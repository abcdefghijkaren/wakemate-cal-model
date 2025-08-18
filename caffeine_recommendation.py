# caffeine_recommendation.py
import numpy as np
from psycopg2.extras import execute_values
from datetime import datetime, timedelta, timezone
from database import get_db_connection

def run_caffeine_recommendation(conn):
    cursor = conn.cursor()

    try:
        # 從資料庫讀取數據
        cursor.execute("SELECT user_id, target_start_time, target_end_time FROM users_target_waking_period")
        waking_periods = cursor.fetchall()

        cursor.execute("SELECT user_id, start_time, end_time FROM users_real_sleep_data")
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
            for user_id_sleep, start_time, end_time in sleep_data:
                if user_id == user_id_sleep:
                    # 使用今天的日期
                    today = datetime.today().date()
                    
                    sleep_hour = start_time.hour
                    wake_hour = end_time.hour
                    total_hours = 24
                    t = np.arange(0, total_hours + 1)

                    def sigmoid(x, L=100, x0=14, k=0.2):
                        return L / (1 + np.exp(-k * (x - x0)))

                    P0_values = np.zeros_like(t, dtype=float)
                    awake_flags = np.ones_like(t, dtype=bool)

                    for h in range(24):
                        asleep = (h >= sleep_hour and h < wake_hour) if sleep_hour < wake_hour else (h >= sleep_hour or h < wake_hour)
                        awake_flags[h] = not asleep
                        P0_values[h] = P0_base + sigmoid(h) if not asleep else P0_base

                    g_PD = np.ones_like(t, dtype=float)
                    P_t_caffeine = np.copy(P0_values)
                    intake_schedule = []

                    daily_dose = 0
                    for hour in range(24):
                        if not awake_flags[hour]:
                            continue
                        if P_t_caffeine[hour] > 270:
                            if daily_dose + dose_unit <= max_daily_dose:
                                # 生成完整的日期時間
                                recommended_time = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc) + timedelta(hours=hour)
                                intake_schedule.append((user_id, hour, dose_unit, recommended_time))
                                daily_dose += dose_unit
                                t_0 = hour
                                effect = 1 / (1 + (M_c * dose_unit / 200) * (k_a / (k_a - k_c)) *
                                            (np.exp(-k_c * (t - t_0)) - np.exp(-k_a * (t - t_0))))
                                effect = np.where(t < t_0, 1, effect)
                                g_PD *= effect
                                P_t_caffeine = P0_values * g_PD

                    # 如果沒有攝取建議，則存入 0
                    if not intake_schedule:
                        recommendations.append((user_id, 0, datetime.combine(today, datetime.min.time())))
                    else:
                        for user_id, _, dose, recommended_time in intake_schedule:
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