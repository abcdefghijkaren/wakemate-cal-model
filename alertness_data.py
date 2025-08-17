# 清醒度數據計算邏輯 alertness_data.py

import numpy as np
from datetime import timedelta
from database import get_db_connection

def fetch_alertness_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM alertness_data_for_visualization;")
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
    except Exception as e:
        print(f"查詢 alertness_data_for_visualization 失敗: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

    if not rows:
        return []

    # 將資料轉換為字典列表
    data = [dict(zip(colnames, row)) for row in rows]
    return data

def run_alertness_data(conn):
    cursor = conn.cursor()

    try:
        # 獲取咖啡因攝取數據
        cursor.execute("SELECT * FROM users_real_time_intake;")
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        caffeine_data = [dict(zip(colnames, row)) for row in rows]

        # 獲取睡眠數據
        cursor.execute("SELECT * FROM users_real_sleep_data;")
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        sleep_data = [dict(zip(colnames, row)) for row in rows]

        # 獲取清醒時段數據
        cursor.execute("SELECT * FROM users_target_waking_period;")
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        target_data = [dict(zip(colnames, row)) for row in rows]

        # 設定參數
        M_c = 1.1
        k_a = 1.0
        k_c = 0.5
        P0_base = 270

        def sigmoid(x, L=100, x0=14, k=0.2):
            return L / (1 + np.exp(-k * (x - x0)))

        # 確保有足夠的數據進行計算
        if not sleep_data or not target_data:
            print("缺少睡眠數據或清醒時段數據，無法計算。")
            return

        # 設定計算的時間範圍
        start_time = min(sleep_data[0]["start_time"], target_data[0]["target_start_time"]).replace(minute=0, second=0)
        end_time = max(sleep_data[0]["end_time"], target_data[0]["target_end_time"]).replace(minute=0, second=0) + timedelta(hours=1)
        total_hours = int((end_time - start_time).total_seconds() // 3600)
        time_index = [start_time + timedelta(hours=i) for i in range(total_hours + 1)]
        t = np.arange(total_hours + 1)

        awake_flags = np.ones(len(time_index), dtype=bool)
        P0_values = np.zeros(len(time_index), dtype=float)

        for i, time in enumerate(time_index):
            is_awake = True
            for row in sleep_data:
                if row["start_time"] <= time < row["end_time"]:
                    is_awake = False
                    break
            awake_flags[i] = is_awake
            hour = time.hour
            P0_values[i] = P0_base + sigmoid(hour) if is_awake else P0_base

        g_PD = np.ones(len(time_index), dtype=float)

        # 如果沒有咖啡因數據，則默認攝取量為 0
        if not caffeine_data:
            caffeine_data = [{'taking_timestamp': start_time, 'caffeine_amount': 0, 'user_id': target_data[0]['user_id']}]

        for row in caffeine_data:
            take_time = row["taking_timestamp"]
            dose = float(row["caffeine_amount"])
            t_0 = int((take_time - start_time).total_seconds() // 3600)
            if t_0 >= len(t):
                continue
            effect = 1 / (1 + (M_c * dose / 200) * (k_a / (k_a - k_c)) *
                          (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
            effect = np.where(t < t_0, 1, effect)
            g_PD *= effect

        P_t_caffeine = P0_values * g_PD

        g_PD_real = np.ones(len(time_index), dtype=float)
        for row in caffeine_data:
            take_time = row["taking_timestamp"]
            dose = float(row["caffeine_amount"])
            t_0 = int((take_time - start_time).total_seconds() // 3600)
            if t_0 >= len(t):
                continue
            effect = 1 / (1 + (M_c * dose / 200) * (k_a / (k_a - k_c)) *
                          (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
            effect = np.where(t < t_0, 1, effect)
            g_PD_real *= effect

        P_t_real = P0_values * g_PD_real
        P_t_no_caffeine = P0_values.copy()

        P_t_caffeine[~awake_flags] = np.nan
        P_t_no_caffeine[~awake_flags] = np.nan
        P_t_real[~awake_flags] = np.nan

        for i in range(len(time_index)):
            cursor.execute("""
                INSERT INTO alertness_data_for_visualization (user_id, timestamp, awake, "g_PD", "P0_values", "P_t_caffeine", "P_t_no_caffeine", "P_t_real")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                caffeine_data[0]['user_id'],
                time_index[i],
                awake_flags[i],
                float(g_PD[i]),
                float(P0_values[i]),
                float(P_t_caffeine[i]),
                float(P_t_no_caffeine[i]),
                float(P_t_real[i])
            ))

        conn.commit()
    except Exception as e:
        print(f"執行清醒度數據計算時發生錯誤: {e}")
    finally:
        cursor.close()