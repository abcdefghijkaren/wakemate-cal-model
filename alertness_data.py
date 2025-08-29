# alertness_data.py
import numpy as np
from datetime import timedelta
from psycopg2.extras import execute_values
from typing import List, Dict

def _get_user_ids_for_alertness(cursor):
    cursor.execute("""
        SELECT DISTINCT user_id FROM (
            SELECT user_id FROM users_target_waking_period
            UNION
            SELECT user_id FROM users_real_sleep_data
            UNION
            SELECT user_id FROM users_real_time_intake
        ) AS u
    """)
    return [row[0] for row in cursor.fetchall()]

def _get_latest_source_ts(cursor, user_id):
    cursor.execute("""
        SELECT GREATEST(
            COALESCE((SELECT MAX(updated_at) FROM users_target_waking_period WHERE user_id = %s), to_timestamp(0)),
            COALESCE((SELECT MAX(updated_at) FROM users_real_sleep_data    WHERE user_id = %s), to_timestamp(0)),
            COALESCE((SELECT MAX(updated_at) FROM users_real_time_intake  WHERE user_id = %s), to_timestamp(0))
        )
    """, (user_id, user_id, user_id))
    (ts,) = cursor.fetchone()
    return ts

def _get_last_processed_ts_for_alert(cursor, user_id):
    cursor.execute("""
        SELECT COALESCE(MAX(source_data_latest_at), to_timestamp(0))
        FROM alertness_data_for_visualization
        WHERE user_id = %s
    """, (user_id,))
    (ts,) = cursor.fetchone()
    return ts

def run_alertness_data(conn):
    cur = conn.cursor()
    try:
        user_ids = _get_user_ids_for_alertness(cur)
        if not user_ids:
            print("缺少必要的輸入資料，無法計算清醒度。")
            return

        # 參數
        M_c = 1.1
        k_a = 1.0
        k_c = 0.5
        P0_base = 270.0  # 固定輸出基準

        def sigmoid(x, L=100, x0=14, k=0.2):
            return L / (1 + np.exp(-k * (x - x0)))

        def safe_float(val, default=0.0):
            try:
                return float(val)
            except Exception:
                return default

        for uid in user_ids:
            latest_source_ts = _get_latest_source_ts(cur, uid)
            last_processed_ts = _get_last_processed_ts_for_alert(cur, uid)

            if latest_source_ts <= last_processed_ts:
                continue

            # 取各表資料
            cur.execute("""
                SELECT sleep_start_time, sleep_end_time
                FROM users_real_sleep_data
                WHERE user_id = %s
                ORDER BY sleep_start_time
            """, (uid,))
            sleep_rows = cur.fetchall()

            cur.execute("""
                SELECT target_start_time, target_end_time
                FROM users_target_waking_period
                WHERE user_id = %s
                ORDER BY target_start_time
            """, (uid,))
            target_rows = cur.fetchall()

            if not sleep_rows or not target_rows:
                continue

            cur.execute("""
                SELECT taking_timestamp, caffeine_amount
                FROM users_real_time_intake
                WHERE user_id = %s
                ORDER BY taking_timestamp
            """, (uid,))
            caf_rows = cur.fetchall()

            cur.execute("""
                SELECT recommended_caffeine_amount, recommended_caffeine_intake_timing
                FROM recommendations_caffeine
                WHERE user_id = %s
                ORDER BY recommended_caffeine_intake_timing
            """, (uid,))
            rec_rows = cur.fetchall()

            # 計算時間範圍
            min_start = min(
                min(r[0] for r in sleep_rows),
                min(r[0] for r in target_rows)
            ).replace(minute=0, second=0, microsecond=0)

            max_end = max(
                max(r[1] for r in sleep_rows),
                max(r[1] for r in target_rows)
            ).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

            total_hours = int((max_end - min_start).total_seconds() // 3600)
            if total_hours < 0:
                continue

            time_index = [min_start + timedelta(hours=i) for i in range(total_hours + 1)]
            t = np.arange(total_hours + 1)

            # 計算 awake_flags
            awake_flags = np.ones(len(time_index), dtype=bool)
            for i, now_dt in enumerate(time_index):
                asleep = any(start <= now_dt < end for (start, end) in sleep_rows)
                awake_flags[i] = (not asleep)

            # ---------- P_t_no_caffeine (自然清醒度) ----------
            P_t_no_caffeine = np.zeros(len(time_index), dtype=float)
            for i, now_dt in enumerate(time_index):
                hour = now_dt.hour
                P_t_no_caffeine[i] = P0_base + sigmoid(hour) if awake_flags[i] else P0_base

            # ---------- P0_values (固定 270) ----------
            P0_values = np.full(len(time_index), P0_base, dtype=float)

            # ---------- g_PD_real ----------
            g_PD_real = np.ones(len(time_index), dtype=float)
            if caf_rows:
                for take_time, dose in caf_rows:
                    dose = safe_float(dose, 0.0)
                    t_0 = int((take_time - min_start).total_seconds() // 3600)
                    if t_0 >= len(t) or t_0 < -10000:
                        continue
                    effect = 1 / (1 + (M_c * dose / 200) * (k_a / (k_a - k_c)) *
                                  (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
                    effect = np.where(t < t_0, 1.0, effect)
                    g_PD_real *= effect

            # ---------- g_PD_rec ----------
            g_PD_rec = np.ones(len(time_index), dtype=float)
            if rec_rows:
                for rec_amount, rec_time in rec_rows:
                    amt = safe_float(rec_amount, 0.0)
                    t_0 = int((rec_time - min_start).total_seconds() // 3600)
                    if t_0 >= len(t) or t_0 < -10000:
                        continue
                    effect = 1 / (1 + (M_c * amt / 200) * (k_a / (k_a - k_c)) *
                                  (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
                    effect = np.where(t < t_0, 1.0, effect)
                    g_PD_rec *= effect

            # ---------- P_t 計算 ----------
            P_t_caffeine = P_t_no_caffeine * g_PD_rec
            P_t_real = P_t_no_caffeine * g_PD_real

            # 睡眠時間改為 0.0
            for arr in (P_t_caffeine, P_t_no_caffeine, P_t_real):
                arr[~awake_flags] = 0.0

            # 設置睡眠時間為 NULL（NaN）
            for arr, g_arr in ((P_t_caffeine, g_PD_rec), (P_t_no_caffeine, g_PD_rec), (P_t_real, g_PD_real)):
                arr[~awake_flags] = np.nan

            # 刪除舊 snapshot
            cur.execute("""
                DELETE FROM alertness_data_for_visualization
                WHERE user_id = %s
                  AND (source_data_latest_at IS NULL OR source_data_latest_at < %s)
            """, (uid, latest_source_ts))

            # 插入資料庫
            insert_rows = []
            for i, now_dt in enumerate(time_index):
                insert_rows.append((
                    uid,
                    now_dt,
                    bool(awake_flags[i]),
                    float(g_PD_rec[i]),
                    float(g_PD_real[i]),
                    float(P0_values[i]),
                    float(P_t_caffeine[i]) if np.isfinite(P_t_caffeine[i]) else None,
                    float(P_t_no_caffeine[i]) if np.isfinite(P_t_no_caffeine[i]) else None,
                    float(P_t_real[i]) if np.isfinite(P_t_real[i]) else None,
                    latest_source_ts
                ))

            execute_values(cur, """
                INSERT INTO alertness_data_for_visualization
                (user_id, timestamp, awake, g_PD_rec, g_PD_real, P0_values, P_t_caffeine, P_t_no_caffeine, P_t_real, source_data_latest_at)
                VALUES %s
            """, insert_rows)

            conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"執行清醒度數據計算時發生錯誤: {e}")
    finally:
        cur.close()