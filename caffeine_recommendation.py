# caffeine_recommendation.py
import numpy as np
from psycopg2.extras import execute_values
from datetime import timedelta
from typing import Optional, Tuple
from user_params import get_user_params  # 讀取個人化參數（若無則回傳預設值）

ALERTNESS_THRESHOLD = 270.0  # 目標上限（ms）
FORBIDDEN_HOURS_BEFORE_SLEEP = 6  # 睡前禁用時數
DOSE_STEP = 25.0              # 劑量單位（四捨五入到 25mg）
MAX_DAILY_DOSE = 300.0        # 一天上限（mg）
WINDOW_HOURS = 2              # 劑量能保持清醒的時長（以小時數計，從攝取後第1小時起算）


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


def _sigmoid(x: float, L: float = 100.0, x0: float = 14.0, k: float = 0.2) -> float:
    return L / (1 + np.exp(-k * (x - x0)))


def _is_in_forbidden_window(ts, sleep_intervals) -> bool:
    """是否位於任何一段睡眠開始前 N 小時的禁用區間。"""
    for (sleep_start, _sleep_end) in sleep_intervals:
        if (sleep_start - timedelta(hours=FORBIDDEN_HOURS_BEFORE_SLEEP)) <= ts < sleep_start:
            return True
    return False


def _compute_precise_dose_for_hour(
    hour_idx: int,
    P0_values: np.ndarray,
    g_PD_current: np.ndarray,
    M_c: float,
    k_a: float,
    k_c: float,
    window_hours: int = WINDOW_HOURS,
    threshold: float = ALERTNESS_THRESHOLD
) -> float:
    """
    反解所需劑量：在投藥後的 1..window_hours 小時內，讓
    P(t) = P0_values[t] * g_PD_current[t] * effect_t(dose) <= threshold
    都成立。回傳滿足所有時點的最小劑量（mg）。
    """
    # 會用到的常數：effect_t(d) = 1 / (1 + A_t * d)，其中
    # A_t = (M_c/200) * (k_a/(k_a - k_c)) * (exp(-k_c*Δ) - exp(-k_a*Δ)), Δ = t - hour_idx
    # 約束：1 / (1 + A_t * d) <= threshold / (P0 * g_PD)  → d >= (1/R - 1)/A_t
    # 注意 Δ=0 時 A_t=0（無立即效應），所以從 Δ=1 起算
    best_required = 0.0
    t_len = len(P0_values)

    for offset in range(1, window_hours + 1):
        t_j = hour_idx + offset
        if t_j >= t_len:
            break

        base = P0_values[t_j] * g_PD_current[t_j]
        if base <= 0:
            continue  # 沒需求

        R = threshold / base
        if R >= 1.0:
            # 本來就 <= threshold，不需藥
            continue

        delta = float(offset)
        phi = np.exp(-k_c * delta) - np.exp(-k_a * delta)
        if phi <= 0:
            # 極少見情況（或數值誤差），略過此點
            continue

        A_t = (M_c / 200.0) * (k_a / (k_a - k_c)) * phi
        # d >= (1/R - 1) / A_t
        required = (1.0 / R - 1.0) / A_t
        if required > best_required:
            best_required = required

    # 劑量粒度四捨五入到 DOSE_STEP
    if best_required <= 0:
        return 0.0
    # 以「無條件進位」比較保守（確保壓住門檻）
    steps = int(np.ceil(best_required / DOSE_STEP))
    return steps * DOSE_STEP


def _apply_dose_to_gpd(
    g_PD: np.ndarray,
    dose_mg: float,
    hour_idx: int,
    M_c: float,
    k_a: float,
    k_c: float
) -> None:
    """把劑量作用乘到 g_PD（就地更新）。"""
    if dose_mg <= 0:
        return
    t = np.arange(len(g_PD))
    t0 = hour_idx
    # 向量化效應
    effect = 1.0 / (1.0 + (M_c * dose_mg / 200.0) * (k_a / (k_a - k_c)) *
                    (np.exp(-k_c * (t - t0)) - np.exp(-k_a * (t - t0))))
    effect = np.where(t < t0, 1.0, effect)  # 投藥前不生效
    g_PD *= effect


def run_caffeine_recommendation(conn):
    cur = conn.cursor()
    try:
        user_ids = _get_distinct_user_ids(cur)
        if not user_ids:
            print("沒有可處理的使用者（沒有清醒/睡眠資料）")
            return

        for uid in user_ids:
            latest_source_ts = _get_latest_source_ts(cur, uid)
            last_processed_ts = _get_last_processed_ts_for_rec(cur, uid)
            if latest_source_ts <= last_processed_ts:
                continue

            # 個人化參數（若 users_params 沒紀錄則回傳預設值）
            M_c, k_a, k_c = get_user_params(cur, uid, default_M_c=1.1, default_k_a=1.0, default_k_c=0.5)

            # 取清醒/睡眠資料
            cur.execute("""
                SELECT target_start_time, target_end_time
                FROM users_target_waking_period
                WHERE user_id = %s
                ORDER BY target_start_time
            """, (uid,))
            waking_periods = cur.fetchall()

            cur.execute("""
                SELECT sleep_start_time, sleep_end_time
                FROM users_real_sleep_data
                WHERE user_id = %s
                ORDER BY sleep_start_time
            """, (uid,))
            sleep_rows = cur.fetchall()

            if not waking_periods or not sleep_rows:
                continue

            # 睡眠區間
            sleep_intervals = [(r[0], r[1]) for r in sleep_rows]

            all_recs = []

            for (target_start_time, target_end_time) in waking_periods:
                # 建立 0..24 小時座標（同原始設計：以一天 24h 掃描）
                total_hours = 24
                t = np.arange(0, total_hours + 1)

                # 基線（清醒→ 270 + sigmoid(h)；睡眠→270）
                P0_values = np.zeros_like(t, dtype=float)
                awake_flags = np.ones_like(t, dtype=bool)

                for h in range(24):
                    now_dt = target_start_time.replace(hour=h, minute=0, second=0, microsecond=0)
                    asleep = any(start <= now_dt < end for (start, end) in sleep_intervals)
                    awake_flags[h] = (not asleep)
                    P0_values[h] = (270.0 + _sigmoid(h)) if (not asleep) else 270.0

                # 當前累積藥效
                g_PD = np.ones_like(t, dtype=float)
                P_t = P0_values * g_PD

                daily_dose = 0.0
                intake_schedule = []

                for hour in range(24):
                    recommended_time = target_start_time.replace(hour=hour, minute=0, second=0, microsecond=0)

                    if _is_in_forbidden_window(recommended_time, sleep_intervals):
                        continue
                    if not awake_flags[hour]:
                        continue
                    # 只在目標區間內做建議
                    if not (target_start_time <= recommended_time <= target_end_time):
                        continue

                    # 若目前預測已 <= 閾值，不必投藥
                    if P_t[hour] <= ALERTNESS_THRESHOLD:
                        continue

                    # 反解「最低需要劑量」
                    dose_needed = _compute_precise_dose_for_hour(
                        hour_idx=hour,
                        P0_values=P0_values,
                        g_PD_current=g_PD,
                        M_c=M_c, k_a=k_a, k_c=k_c,
                        window_hours=WINDOW_HOURS,
                        threshold=ALERTNESS_THRESHOLD
                    )

                    # 受每日上限限制
                    remaining = MAX_DAILY_DOSE - daily_dose
                    dose_to_give = min(dose_needed, max(0.0, remaining))
                    # 若需要量非常小（或剩餘為 0），就略過
                    if dose_to_give <= 0:
                        continue

                    # 記錄這次投藥
                    intake_schedule.append((uid, dose_to_give, recommended_time))
                    daily_dose += dose_to_give

                    # 把劑量作用進目前的 g_PD，並更新 P_t 供後續小時判斷使用
                    _apply_dose_to_gpd(g_PD, dose_to_give, hour, M_c, k_a, k_c)
                    P_t = P0_values * g_PD

                if intake_schedule:
                    all_recs.extend(intake_schedule)

            if all_recs:
                # 清掉較舊的建議（可選）
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
                    [(r[0], r[1], r[2], latest_source_ts) for r in all_recs]
                )
                conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] 執行咖啡因建議計算時發生錯誤: {e}")
    finally:
        cur.close()