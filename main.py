# 主執行檔，包含 run_caffeine_recommendation 和 run_alertness_data 的調用
import time
import os
from database import get_db_connection
from caffeine_recommendation import run_caffeine_recommendation
from alertness_data import run_alertness_data

def main():
    while True:
        # 連接到資料庫
        conn = get_db_connection()
        
        # 執行咖啡因建議計算
        run_caffeine_recommendation(conn)
        
        # 執行清醒度數據計算
        run_alertness_data(conn)
        
        # 每隔一段時間執行一次，先設定每小時執行一次
        time.sleep(3600)  # 3600秒 = 1小時

if __name__ == "__main__":
    main()
