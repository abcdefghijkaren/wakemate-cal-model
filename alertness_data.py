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
        P0_base = 270.0  # 固定基線 (ms)

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

            # 沒新資料 → 跳過
            if latest_source_ts <= last_processed_ts:
                continue

            # 取各表資料（單一使用者）
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
                # 兩者皆需，否則無法得出合理範圍
                continue

            # 取實際攝取（users_real_time_intake）
            cur.execute("""
                SELECT taking_timestamp, caffeine_amount
                FROM users_real_time_intake
                WHERE user_id = %s
                ORDER BY taking_timestamp
            """, (uid,))
            caf_rows = cur.fetchall()  # list of (taking_timestamp, caffeine_amount)

            # 取建議攝取（recommendations_caffeine）作為 "recommended schedule"
            # 注意：recommendations_caffeine 的 recommended_caffeine_intake_timing 須為 timestamp
            cur.execute("""
                SELECT recommended_caffeine_amount, recommended_caffeine_intake_timing
                FROM recommendations_caffeine
                WHERE user_id = %s
                ORDER BY recommended_caffeine_intake_timing
            """, (uid,))
            rec_rows = cur.fetchall()  # list of (amount, datetime)

            # 設定計算時間範圍（以 sleep 與 target 的 min/max 為準）
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

            # 計算每個小時是否清醒 & P0_values 固定 270
            awake_flags = np.ones(len(time_index), dtype=bool)
            P0_values = np.full(len(time_index), P0_base, dtype=float)

            for i, now_dt in enumerate(time_index):
                asleep = any(start <= now_dt < end for (start, end) in sleep_rows)
                awake_flags[i] = (not asleep)

            # ---------- 計算 g_PD_real（使用者真實攝取） ----------
            g_PD_real = np.ones(len(time_index), dtype=float)
            if caf_rows:
                for take_time, dose in caf_rows:
                    dose = safe_float(dose, 0.0)
                    # 計算相對小時索引
                    t_0 = int((take_time - min_start).total_seconds() // 3600)
                    # 若 t_0 超出範圍就跳過
                    if t_0 >= len(t) or t_0 < -10000:  # 大幅負數視為不合理，保險處理
                        continue
                    # effect 的向量計算（同 formula）
                    effect = 1 / (1 + (M_c * dose / 200) * (k_a / (k_a - k_c)) *
                                  (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
                    # 在 t < t_0 時設定 effect=1（尚未起效）
                    effect = np.where(t < t_0, 1.0, effect)
                    g_PD_real *= effect

            # ---------- 計算 g_PD_rec（建議攝取） ----------
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

            # P_t 計算
            P_t_no_caffeine = np.copy(P0_values)             # 若完全不攝取
            P_t_caffeine = P0_values * g_PD_rec              # 若依照建議攝取
            P_t_real = P0_values * g_PD_real                 # 實際攝取情況

            # 睡覺時段值改成 0.0（避免 NULL，且表明非觀察）
            for arr in (P_t_caffeine, P_t_no_caffeine, P_t_real):
                arr[~awake_flags] = 0.0

            # 刪除舊 snapshot（若存在較舊的 source_data_latest_at）
            cur.execute("""
                DELETE FROM alertness_data_for_visualization
                WHERE user_id = %s
                  AND (source_data_latest_at IS NULL OR source_data_latest_at < %s)
            """, (uid, latest_source_ts))

            # 設置睡眠時段值為 NULL
            for arr, g_arr in ((P_t_caffeine, g_PD_rec), (P_t_no_caffeine, g_PD_rec), (P_t_real, g_PD_real)):
                arr[~awake_flags] = np.nan  # NaN 會被 psycopg2 轉成 NULL

            # 準備插入
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
                (user_id, timestamp, awake, g_PD_rec, g_PD_real, P0_values, P_t_caffeine, P_t_no_caffeine, P_t_real, updated_at)
                VALUES %s
            """, insert_rows)

            conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"執行清醒度數據計算時發生錯誤: {e}")
    finally:
        cur.close()