import sqlite3

# Replace with your database file path
db_path = 'trading_algo.db'

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Query to list all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Print table names
print("Tables in the database:")
for table in tables:
    print(table[0])

# Close the connection
conn.close()