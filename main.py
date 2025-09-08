# 主執行檔，包含 run_caffeine_recommendation 和 run_alertness_data 的調用
# main.py
from database import get_db_connection
from caffeine_recommendation import run_caffeine_recommendation
from alertness_data import run_alertness_data

def get_user_params(conn):
    """
    從 users_params 撈取所有使用者的 M_c, k_a, k_c
    回傳 dict: { user_id: {"M_c": x, "k_a": y, "k_c": z}, ... }
    """
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT user_id, M_c, k_a, k_c
            FROM users_params
        """)
        rows = cur.fetchall()
        params_map = {}
        for user_id, M_c, k_a, k_c in rows:
            params_map[user_id] = {
                "M_c": M_c,
                "k_a": k_a,
                "k_c": k_c
            }
        return params_map
    finally:
        cur.close()

def main():
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            print("無法連到資料庫")
            return

        try:
            # 取所有使用者的個人化參數
            user_params_map = get_user_params(conn)

            # 先跑咖啡因建議（傳入個人化參數）
            run_caffeine_recommendation(conn, user_params_map)

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
