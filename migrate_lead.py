import psycopg2

def migrate_lead():
    conn_str = "postgres://postgres.ulpttibobjmzctrgiiey:SdrWhastapp2026Base@aws-0-ca-central-1.pooler.supabase.com:6543/postgres"
    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        cur.execute("UPDATE leads SET company_id = 'rayccs-gmail-com' WHERE id = 21")
        conn.commit()
        print("Updated lead 21 to rayccs-gmail-com")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    migrate_lead()
