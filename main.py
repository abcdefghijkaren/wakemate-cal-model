# 主執行檔，包含 run_caffeine_recommendation 和 run_alertness_data 的調用
# main.py
from database import get_db_connection
from caffeine_recommendation import run_caffeine_recommendation
from alertness_data import run_alertness_data

def main():
    # 連資料庫
    try:
        conn = get_db_connection()
        if conn is None:
            print("無法連到資料庫")
            return
        
        try:
            # 執行咖啡因建議計算
            run_caffeine_recommendation(conn)
            # 執行清醒度數據計算
            run_alertness_data(conn)
        except Exception as e:
            print(f"[ERROR] 在執行計算時發生錯誤: {e}")
        
    except Exception as e:
        print(f"[ERROR] 連接資料庫時發生錯誤: {e}")
    finally:
        # 確保關閉連線
        if 'conn' in locals() and conn is not None:
            conn.close()

if __name__ == "__main__":
    main()