import numpy as np
from psycopg2.extras import execute_values
from datetime import datetime, timedelta, timezone
from database import get_db_connection

def run_caffeine_recommendation(conn):
    cursor = conn.cursor()

    try:
        # 從資料庫讀取數據
        cursor.execute("SELECT user_id, target_start_time AT TIME ZONE 'UTC', target_end_time AT TIME ZONE 'UTC' FROM users_target_waking_period")
        waking_periods = cursor.fetchall()

        cursor.execute("SELECT user_id, sleep_start_time AT TIME ZONE 'UTC', sleep_end_time AT TIME ZONE 'UTC' FROM users_real_sleep_data")
        sleep_data = cursor.fetchall()

        # 檢查是否有足夠的數據
        if not waking_periods:
            print("警告: 沒有找到清醒時段數據。")
            return
        if not sleep_data:
            print("警告: 沒有找到睡眠數據。")
            return

        # 參數設定
        M_c = 1.1
        k_a = 1.0
        k_c = 0.5
        P0_base = 270
        dose_unit = 100
        max_daily_dose = 300

        # 模擬咖啡因影響與攝取建議
        recommendations = []

        for user_id, target_start_time, target_end_time in waking_periods:
            # 獲取該使用者的睡眠數據
            user_sleep_data = [s for s in sleep_data if s[0] == user_id]
            if not user_sleep_data:
                recommendations.append((user_id, 0, target_start_time))  # 如果沒有睡眠數據，則存入0
                continue

            for user_id, sleep_start_time, sleep_end_time in user_sleep_data:
                # 計算睡眠時間的前六小時
                sleep_start = sleep_start_time - timedelta(hours=6)

                # 設定時間範圍
                total_hours = 24
                t = np.arange(0, total_hours + 1)

                def sigmoid(x, L=100, x0=14, k=0.2):
                    return L / (1 + np.exp(-k * (x - x0)))

                P0_values = np.zeros_like(t, dtype=float)
                awake_flags = np.ones_like(t, dtype=bool)

                for h in range(24):
                    asleep = (h >= sleep_start_time.hour and h < sleep_end_time.hour) if sleep_start_time.hour < sleep_end_time.hour else (h >= sleep_start_time.hour or h < sleep_end_time.hour)
                    awake_flags[h] = not asleep
                    P0_values[h] = P0_base + sigmoid(h) if not asleep else P0_base

                g_PD = np.ones_like(t, dtype=float)
                P_t_caffeine = np.copy(P0_values)
                intake_schedule = []

                daily_dose = 0
                for hour in range(24):
                    # 生成完整的日期時間
                    recommended_time = target_start_time.replace(hour=hour, minute=0, second=0, microsecond=0)

                    # 檢查是否在睡眠前六小時內
                    if recommended_time < sleep_start:
                        if not awake_flags[hour]:
                            continue
                        if P_t_caffeine[hour] > 270:
                            if daily_dose + dose_unit <= max_daily_dose:
                                intake_schedule.append((user_id, dose_unit, recommended_time))
                                daily_dose += dose_unit
                                t_0 = hour
                                effect = 1 / (1 + (M_c * dose_unit / 200) * (k_a / (k_a - k_c)) *
                                               (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
                                effect = np.where(t < t_0, 1, effect)
                                g_PD *= effect
                                P_t_caffeine = P0_values * g_PD

                # 檢查推薦時間是否在 target_start_time 和 target_end_time 之間
                for user_id, dose, recommended_time in intake_schedule:
                    if target_start_time <= recommended_time <= target_end_time:
                        recommendations.append((user_id, dose, recommended_time))

        # 插入建議到資料庫
        if recommendations:
            execute_values(cursor, 
                "INSERT INTO recommendations_caffeine (user_id, recommended_caffeine_amount, recommended_caffeine_intake_timing) VALUES %s",
                [(r[0], r[1], r[2]) for r in recommendations])
            conn.commit()
        else:
            print("沒有建議需要插入到資料庫。")

    except Exception as e:
        print(f"執行咖啡因建議計算時發生錯誤: {e}")
    finally:
        cursor.close()
