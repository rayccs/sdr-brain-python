import psycopg2

def query_lead():
    conn_str = "postgres://postgres.ulpttibobjmzctrgiiey:SdrWhastapp2026Base@aws-0-ca-central-1.pooler.supabase.com:6543/postgres"
    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT company_id FROM leads")
        rows = cur.fetchall()
        for r in rows:
            print(r)
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    query_lead()
