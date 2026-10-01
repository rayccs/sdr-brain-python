import psycopg2

def query_lead():
    conn_str = "postgres://postgres.ulpttibobjmzctrgiiey:SdrWhastapp2026Base@aws-0-ca-central-1.pooler.supabase.com:6543/postgres"
    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        cur.execute("SELECT id, phone, name, status, company_id FROM leads WHERE company_id = 'rayccs-gmail-com' ORDER BY updated_at DESC")
        rows = cur.fetchall()
        print(f'Found {len(rows)} leads')
        for r in rows:
            print(r)
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    query_lead()
