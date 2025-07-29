# 主執行檔，包含 run_caffeine_recommendation 和 run_alertness_data 的調用
# main.py
from database import get_db_connection
from caffeine_recommendation import run_caffeine_recommendation
from alertness_data import run_alertness_data

def main():
    # 連接到資料庫
    conn = get_db_connection()
    
    # 執行咖啡因建議計算
    run_caffeine_recommendation(conn)
    
    # 執行清醒度數據計算
    run_alertness_data(conn)
    
    # 關閉連線（如果你有這個功能的話）
    conn.close()

if __name__ == "__main__":
    main()