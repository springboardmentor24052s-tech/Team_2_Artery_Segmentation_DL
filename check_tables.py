import psycopg
conn = psycopg.connect('postgresql://postgres:8727@localhost:5432/artery_db')
cur = conn.cursor()
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
tables = cur.fetchall()
print('Tables:', [t[0] for t in tables])
conn.close()