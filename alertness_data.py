# 清醒度數據計算邏輯 alertness_data.py

import pandas as pd
import numpy as np
from datetime import timedelta
from database import get_db_connection

def fetch_alertness_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alertness_data;")  # 變更
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    cursor.close()
    conn.close()
    df = pd.DataFrame(rows, columns=colnames)
    if df.empty:
        return pd.DataFrame()
    df = df.drop_duplicates(subset=["user_id", "timestamp"])
    return df

def run_alertness_data(conn):
    cursor = conn.cursor()

    # 變更：改成 fetchall() 再 DataFrame
    cursor.execute("SELECT * FROM users_real_time_intake;")
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    caffeine_df = pd.DataFrame(rows, columns=colnames)
    caffeine_df["taking_timestamp"] = pd.to_datetime(caffeine_df["taking_timestamp"])

    cursor.execute("SELECT * FROM users_real_sleep_data;")
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    sleep_df = pd.DataFrame(rows, columns=colnames)
    sleep_df["start_time"] = pd.to_datetime(sleep_df["start_time"])
    sleep_df["end_time"] = pd.to_datetime(sleep_df["end_time"])

    cursor.execute("SELECT * FROM users_target_waking_period;")
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    target_df = pd.DataFrame(rows, columns=colnames)
    target_df["target_start_time"] = pd.to_datetime(target_df["target_start_time"])
    target_df["target_end_time"] = pd.to_datetime(target_df["target_end_time"])

    M_c = 1.1
    k_a = 1.0
    k_c = 0.5
    P0_base = 270

    def sigmoid(x, L=100, x0=14, k=0.2):
        return L / (1 + np.exp(-k * (x - x0)))

    start_time = min(caffeine_df["taking_timestamp"].min(), sleep_df["start_time"].min()).replace(minute=0, second=0)
    end_time = max(caffeine_df["taking_timestamp"].max(), sleep_df["end_time"].max()).replace(minute=0, second=0) + timedelta(hours=1)
    total_hours = int((end_time - start_time).total_seconds() // 3600)
    time_index = [start_time + timedelta(hours=i) for i in range(total_hours + 1)]
    t = np.arange(total_hours + 1)

    awake_flags = np.ones(len(time_index), dtype=bool)
    P0_values = np.zeros(len(time_index), dtype=float)

    for i, time in enumerate(time_index):
        is_awake = True
        for _, row in sleep_df.iterrows():
            if row["start_time"] <= time < row["end_time"]:
                is_awake = False
                break
        awake_flags[i] = is_awake
        hour = time.hour
        P0_values[i] = P0_base + sigmoid(hour) if is_awake else P0_base

    g_PD = np.ones(len(time_index), dtype=float)
    for _, row in caffeine_df.iterrows():
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
    for _, row in caffeine_df.iterrows():
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
            caffeine_df['user_id'].iloc[0],
            time_index[i],
            awake_flags[i].item(),
            float(g_PD[i]),
            float(P0_values[i]),
            float(P_t_caffeine[i]),
            float(P_t_no_caffeine[i]),
            float(P_t_real[i])
        ))

    conn.commit()
    cursor.close()