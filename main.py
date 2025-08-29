# 主執行檔，包含 run_caffeine_recommendation 和 run_alertness_data 的調用
# main.py
from database import get_db_connection
from caffeine_recommendation import run_caffeine_recommendation
from alertness_data import run_alertness_data

def main():
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            print("無法連到資料庫")
            return

        try:
            # 先跑建議（只會對有「新資料」的使用者動作）
            run_caffeine_recommendation(conn)
            # 再跑清醒度（同樣只處理有「新資料」的使用者）
            run_alertness_data(conn)
        except Exception as e:
            print(f"[ERROR] 在執行計算時發生錯誤: {e}")

    except Exception as e:
        print(f"[ERROR] 連接資料庫時發生錯誤: {e}")
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    main()