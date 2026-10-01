import psycopg2
import sys

def query_conversations():
    sys.stdout.reconfigure(encoding='utf-8')
    conn_str = "postgres://postgres.ulpttibobjmzctrgiiey:SdrWhastapp2026Base@aws-0-ca-central-1.pooler.supabase.com:6543/postgres"
    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        cur.execute("SELECT id, role, content FROM conversations WHERE lead_id = 21 ORDER BY created_at ASC")
        rows = cur.fetchall()
        print(f"Found {len(rows)} messages for lead 21")
        for r in rows:
            print(f"[{r[0]}] {r[1]}: {r[2][:100]}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    query_conversations()
