# 資料庫連接和操作的封裝 database.py

import os
import psycopg2
# 獲取環境變數中的資料庫連接字串
DATABASE_URL = os.getenv('DATABASE_URL')
def get_db_connection():
    try:
        connection = psycopg2.connect(DATABASE_URL)
        print("成功連到資料庫")
        return connection
    except Exception as e:
        print(f"連接失敗: {e}")
        return None