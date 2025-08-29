# caffeine_recommendation.py
import numpy as np
from psycopg2.extras import execute_values
from datetime import timedelta
from typing import Optional

def _get_distinct_user_ids(cursor):
    cursor.execute("""
        SELECT DISTINCT user_id FROM (
            SELECT user_id FROM users_target_waking_period
            UNION
            SELECT user_id FROM users_real_sleep_data
        ) AS u
    """)
    return [row[0] for row in cursor.fetchall()]

def _get_latest_source_ts(cursor, user_id):
    """
    回傳此使用者在「清醒時段 + 睡眠 + 咖啡因攝取」三張表的最新 updated_at。
    若某表沒有資料則以 epoch 取代。
    """
    cursor.execute("""
        SELECT GREATEST(
            COALESCE((SELECT MAX(updated_at) FROM users_target_waking_period WHERE user_id = %s), to_timestamp(0)),
            COALESCE((SELECT MAX(updated_at) FROM users_real_sleep_data    WHERE user_id = %s), to_timestamp(0)),
            COALESCE((SELECT MAX(updated_at) FROM users_real_time_intake  WHERE user_id = %s), to_timestamp(0))
        )
    """, (user_id, user_id, user_id))
    (ts,) = cursor.fetchone()
    return ts

def _get_last_processed_ts_for_rec(cursor, user_id):
    cursor.execute("""
        SELECT COALESCE(MAX(source_data_latest_at), to_timestamp(0))
        FROM recommendations_caffeine
        WHERE user_id = %s
    """, (user_id,))
    (ts,) = cursor.fetchone()
    return ts

def run_caffeine_recommendation(conn):
    cur = conn.cursor()
    try:
        user_ids = _get_distinct_user_ids(cur)
        if not user_ids:
            print("沒有可處理的使用者（沒有清醒/睡眠資料）")
            return

        # 參數設定
        M_c = 1.1
        k_a = 1.0
        k_c = 0.5
        P0_base = 270
        dose_unit = 100
        max_daily_dose = 300

        def sigmoid(x, L=100, x0=14, k=0.2):
            return L / (1 + np.exp(-k * (x - x0)))

        for uid in user_ids:
            latest_source_ts = _get_latest_source_ts(cur, uid)
            last_processed_ts = _get_last_processed_ts_for_rec(cur, uid)

            # 沒有新資料就跳過
            if latest_source_ts <= last_processed_ts:
                continue

            # 取該使用者資料
            cur.execute("""
                SELECT user_id, target_start_time, target_end_time
                FROM users_target_waking_period
                WHERE user_id = %s
                ORDER BY target_start_time
            """, (uid,))
            waking_periods = cur.fetchall()

            cur.execute("""
                SELECT user_id, sleep_start_time, sleep_end_time
                FROM users_real_sleep_data
                WHERE user_id = %s
                ORDER BY sleep_start_time
            """, (uid,))
            sleep_rows = cur.fetchall()

            if not waking_periods or not sleep_rows:
                # 缺資料就不要產生建議
                continue

            # 先組 sleep 時段（可能有多段）
            sleep_intervals = [(r[1], r[2]) for r in sleep_rows]  # (start, end)

            recommendations = []

            # 對每個清醒目標區段做建議
            for _, target_start_time, target_end_time in waking_periods:
                total_hours = 24
                t = np.arange(0, total_hours + 1)

                P0_values = np.zeros_like(t, dtype=float)
                awake_flags = np.ones_like(t, dtype=bool)

                for h in range(24):
                    now_dt = target_start_time.replace(hour=h, minute=0, second=0, microsecond=0)
                    # 是否在任何一個睡眠區間內
                    asleep = any(start <= now_dt < end for (start, end) in sleep_intervals)
                    awake_flags[h] = (not asleep)
                    P0_values[h] = P0_base + sigmoid(h) if not asleep else P0_base

                g_PD = np.ones_like(t, dtype=float)
                P_t_caffeine = np.copy(P0_values)

                daily_dose = 0
                intake_schedule = []

                for hour in range(24):
                    recommended_time = target_start_time.replace(hour=hour, minute=0, second=0, microsecond=0)

                    # 禁止睡前 6 小時
                    in_forbidden = any(
                        (sleep_start - timedelta(hours=6)) <= recommended_time < sleep_start
                        for (sleep_start, sleep_end) in sleep_intervals
                    )
                    if in_forbidden:
                        continue

                    if not awake_flags[hour]:
                        continue

                    if P_t_caffeine[hour] > 270 and (daily_dose + dose_unit) <= max_daily_dose:
                        intake_schedule.append((uid, dose_unit, recommended_time))
                        daily_dose += dose_unit
                        t_0 = hour
                        effect = 1 / (1 + (M_c * dose_unit / 200) * (k_a / (k_a - k_c)) *
                                       (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
                        effect = np.where(t < t_0, 1, effect)
                        g_PD *= effect
                        P_t_caffeine = P0_values * g_PD

                # 僅保留落在 target 區間內的建議
                filtered = [
                    (uid, dose, when)
                    for (uid, dose, when) in intake_schedule
                    if target_start_time <= when <= target_end_time
                ]
                recommendations.extend(filtered)

            if recommendations:
                # （可選）清掉更舊的建議，避免表越長越大
                cur.execute("""
                    DELETE FROM recommendations_caffeine
                    WHERE user_id = %s
                      AND (source_data_latest_at IS NULL OR source_data_latest_at < %s)
                """, (uid, latest_source_ts))

                execute_values(
                    cur,
                    """
                    INSERT INTO recommendations_caffeine
                    (user_id, recommended_caffeine_amount, recommended_caffeine_intake_timing, source_data_latest_at)
                    VALUES %s
                    """,
                    [(r[0], r[1], r[2], latest_source_ts) for r in recommendations]
                )
                conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"執行咖啡因建議計算時發生錯誤: {e}")
    finally:
        cur.close()
