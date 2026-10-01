import psycopg2
import sys

def delete_records():
    conn_str = "postgres://postgres.ulpttibobjmzctrgiiey:SdrWhastapp2026Base@aws-0-ca-central-1.pooler.supabase.com:6543/postgres"
    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations WHERE content LIKE '%Actualmente estoy presentando fallas técnicas%'")
        print(f"Deleted {cur.rowcount} error messages from conversations.")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    delete_records()
