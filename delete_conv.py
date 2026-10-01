import psycopg2

def delete_conversations():
    conn_str = "postgres://postgres.ulpttibobjmzctrgiiey:SdrWhastapp2026Base@aws-0-ca-central-1.pooler.supabase.com:6543/postgres"
    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations WHERE lead_id = 21")
        conn.commit()
        print("Deleted conversations for lead 21")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    delete_conversations()
