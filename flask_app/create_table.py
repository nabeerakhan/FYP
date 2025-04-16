import sqlite3

# Connect to the database file (creates it if it doesn't exist)
conn = sqlite3.connect('db/attendance.db')
c = conn.cursor()

# Create the attendance table
c.execute('''CREATE TABLE attendance (
                id INTEGER PRIMARY KEY,
                student_name TEXT,
                emotion TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
             )''')

conn.commit()
conn.close()

print("Table created successfully.")
