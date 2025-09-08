# user_params.py
import numpy as np
from datetime import timedelta

def get_user_params(cur, user_id, default_M_c=1.1, default_k_a=1.0, default_k_c=0.5):
    cur.execute("""
        SELECT M_c, k_a, k_c
        FROM users_params
        WHERE user_id = %s
        ORDER BY updated_at DESC
        LIMIT 1
    """, (user_id,))
    row = cur.fetchone()
    if not row:
        return (default_M_c, default_k_a, default_k_c)
    return (float(row[0]), float(row[1]), float(row[2]))


def upsert_user_params(cur, user_id, M_c, k_a, k_c):
    """
    需要 users_params.user_id 有 UNIQUE constraint（或做 PRIMARY KEY）。
    """
    cur.execute("""
        INSERT INTO users_params (user_id, M_c, k_a, k_c, updated_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (user_id) DO UPDATE
          SET M_c = EXCLUDED.M_c,
              k_a = EXCLUDED.k_a,
              k_c = EXCLUDED.k_c,
              updated_at = NOW()
    """, (user_id, float(M_c), float(k_a), float(k_c)))


def _sigmoid(x: float, L: float = 100.0, x0: float = 14.0, k: float = 0.2) -> float:
    return L / (1 + np.exp(-k * (x - x0)))


def _predict_alertness_at(
    t_obs,
    min_start,
    sleep_intervals,
    intakes,          # list[(take_time, dose)]
    M_c, k_a, k_c
) -> float:
    """
    在時間點 t_obs 預測 P(t)。
    - Baseline（清醒）：270 + sigmoid(hour)
      Baseline（睡眠）：270
    - g_PD_real 由 intakes 累乘
    """
    # baseline
    asleep = any(start <= t_obs < end for (start, end) in sleep_intervals)
    base = 270.0 if asleep else (270.0 + _sigmoid(t_obs.hour))

    # g_PD_real
    g = 1.0
    for (take_time, dose) in intakes:
        if take_time > t_obs:
            continue
        dt_h = (t_obs - take_time).total_seconds() / 3600.0
        if dt_h <= 0:
            continue
        phi = np.exp(-k_c * dt_h) - np.exp(-k_a * dt_h)
        eff = 1.0 / (1.0 + (M_c * dose / 200.0) * (k_a / (k_a - k_c)) * phi)
        g *= eff

    return base * g


def fit_user_params(conn, user_id):
    """
    使用者個人參數回推（粗網格搜尋），無 pandas / scipy。
    需要資料表：
      - users_real_time_intake(user_id, taking_timestamp, caffeine_amount)
      - users_real_sleep_data(user_id, sleep_start_time, sleep_end_time)
      - users_pvt_results(user_id, test_at, mean_rt)  ← 欄位名稱請對照你的實際表
    """
    cur = conn.cursor()
    try:
        # 取實際攝取
        cur.execute("""
            SELECT taking_timestamp, caffeine_amount
            FROM users_real_time_intake
            WHERE user_id = %s
            ORDER BY taking_timestamp
        """, (user_id,))
        intakes = [(r[0], float(r[1])) for r in cur.fetchall()]

        # 取睡眠
        cur.execute("""
            SELECT sleep_start_time, sleep_end_time
            FROM users_real_sleep_data
            WHERE user_id = %s
            ORDER BY sleep_start_time
        """, (user_id,))
        sleep_rows = cur.fetchall()
        if not sleep_rows or not intakes:
            return  # 無足夠資料不校準
        sleep_intervals = [(r[0], r[1]) for r in sleep_rows]

        # 取 PVT
        cur.execute("""
            SELECT test_at, mean_rt
            FROM users_pvt_results
            WHERE user_id = %s
            ORDER BY test_at
        """, (user_id,))
        pvt_rows = cur.fetchall()
        if not pvt_rows:
            return

        # 搜尋範圍（可自行調整）
        M_c_grid = np.arange(0.8, 1.6, 0.1)
        k_a_grid = np.arange(0.6, 1.6, 0.1)
        k_c_grid = np.arange(0.2, 1.1, 0.1)

        best = None
        best_err = float("inf")

        # min_start 用不到（此預測函式以絕對時間算），但保留參數接口
        min_start = min(sleep_intervals[0][0], intakes[0][0])

        for Mc in M_c_grid:
            for ka in k_a_grid:
                for kc in k_c_grid:
                    if ka <= kc + 0.05:
                        continue  # 基本藥代學限制
                    se = 0.0
                    n = 0
                    for (t_obs, y_obs) in pvt_rows:
                        y_hat = _predict_alertness_at(
                            t_obs, min_start, sleep_intervals, intakes, Mc, ka, kc
                        )
                        err = (y_hat - float(y_obs)) ** 2
                        se += err
                        n += 1
                    if n == 0:
                        continue
                    mse = se / n
                    if mse < best_err:
                        best_err = mse
                        best = (Mc, ka, kc)

        if best is not None:
            upsert_user_params(cur, user_id, *best)
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()