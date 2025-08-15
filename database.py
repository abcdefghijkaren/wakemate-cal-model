# 資料庫連接和操作的封裝 database.py
import os
import psycopg2
# 獲取環境變數中的資料庫連接字串
DATABASE_URL = os.getenv('DATABASE_URL')
# 連接到 PostgreSQL 資料庫
try:
    connection = psycopg2.connect(DATABASE_URL)
    cursor = connection.cursor()
    print("成功連接到資料庫")
except Exception as e:
    print(f"連接失敗: {e}")
finally:
    if connection:
        cursor.close()
        connection.close()