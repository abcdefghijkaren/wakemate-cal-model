import numpy as np
from psycopg2.extras import execute_values
from datetime import datetime, timedelta, timezone
from database import get_db_connection

def run_caffeine_recommendation(conn):
    cursor = conn.cursor()

    try:
        # 從資料庫讀取數據，確保時間轉為UTC時區
        cursor.execute("SELECT user_id, target_start_time AT TIME ZONE 'UTC', target_end_time AT TIME ZONE 'UTC' FROM users_target_waking_period")
        waking_periods = cursor.fetchall()

        cursor.execute("SELECT user_id, start_time AT TIME ZONE 'UTC', end_time AT TIME ZONE 'UTC' FROM users_real_sleep_data")
        sleep_data = cursor.fetchall()

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

        recommendations = []
        now = datetime.now(timezone.utc)
        today = now.date()

        for user_id, target_start_time, target_end_time in waking_periods:
            # 找到該使用者的睡眠數據
            user_sleep_periods = [s for s in sleep_data if s[0] == user_id]
            if not user_sleep_periods:
                continue

            # 初始化變數
            awake_hours = set()
            no_caffeine_hours = set()

            # 處理睡眠和前6小時不推薦的時間段
            for sleep_start, sleep_end in user_sleep_periods:
                # 計算睡覺時間和前6小時
                sleep_start = sleep_start.replace(tzinfo=timezone.utc)
                sleep_end = sleep_end.replace(tzinfo=timezone.utc)
                no_caffeine_start = sleep_start - timedelta(hours=6)
                
                # 記錄不推薦時段
                current = no_caffeine_start
                while current < sleep_end:
                    no_caffeine_hours.add((current.hour, current.minute))
                    current += timedelta(minutes=30)

            # 計算清醒時間段
            target_start_time = target_start_time.replace(tzinfo=timezone.utc)
            target_end_time = target_end_time.replace(tzinfo=timezone.utc)
            
            # 模擬計算
            t = np.arange(0, 24)
            P0_values = np.zeros_like(t, dtype=float)
            awake_flags = np.ones_like(t, dtype=bool)

            def sigmoid(x, L=100, x0=14, k=0.2):
                return L / (1 + np.exp(-k * (x - x0)))

            # 計算P0和清醒狀態
            for h in range(24):
                check_time = datetime.combine(today, datetime.min.time()) + timedelta(hours=h)
                check_time = check_time.replace(tzinfo=timezone.utc)
                
                # 檢查是否在睡眠時間或前6小時
                is_sleep_period = False
                for sleep_start, sleep_end in user_sleep_periods:
                    if (sleep_start.time() <= check_time.time() < sleep_end.time() if sleep_start.time() < sleep_end.time() 
                        else check_time.time() >= sleep_start.time() or check_time.time() < sleep_end.time()):
                        is_sleep_period = True
                        break
                
                asleep = is_sleep_period or ((check_time.hour, check_time.minute) in no_caffeine_hours)
                awake_flags[h] = not asleep
                P0_values[h] = P0_base + sigmoid(h) if not asleep else P0_base

            # 模擬咖啡因影響
            g_PD = np.ones_like(t, dtype=float)
            P_t_caffeine = np.copy(P0_values)
            daily_dose = 0

            for h in range(24):
                check_time = datetime.combine(today, datetime.min.time()) + timedelta(hours=h)
                check_time = check_time.replace(tzinfo=timezone.utc)
                
                # 只考慮在目標時段內的時間
                if not (target_start_time.time() <= check_time.time() <= target_end_time.time()):
                    continue

                if not awake_flags[h]:
                    continue

                if P_t_caffeine[h] > 270 and daily_dose + dose_unit <= max_daily_dose:
                    dose_time = check_time
                    recommendations.append((user_id, dose_unit, dose_time))
                    daily_dose += dose_unit
                    
                    # 計算咖啡因影響
                    t_0 = h
                    effect = 1 / (1 + (M_c * dose_unit / 200) * (k_a/(k_a - k_c)) * (np.exp(-k_c*(t - t_0)) - np.exp(-k_a*(t - t_0))))
                    effect = np.where(t < t_0, 1, effect)
                    g_PD *= effect
                    P_t_caffeine = P0_values * g_PD

            # 如果在目標時段內沒有任何推薦，則存入0
            if not any(r[0] == user_id for r in recommendations):
                recommendations.append((user_id, 0, target_start_time))

        # 插入建議到資料庫
        if recommendations:
            execute_values(cursor, 
                """INSERT INTO recommendations_caffeine 
                   (user_id, recommended_caffeine_amount, recommended_caffeine_intake_timing) 
                   VALUES %s 
                   ON CONFLICT (user_id, recommended_caffeine_intake_timing) 
                   DO NOTHING""",
                [(r[0], r[1], r[2]) for r in recommendations])
            conn.commit()
        else:
            print("沒有建議需要插入到資料庫。")

    except Exception as e:
        print(f"執行咖啡因建議計算時發生錯誤: {e}")
        conn.rollback()
    finally:
        cursor.close()
