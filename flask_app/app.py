from flask import Flask, render_template, request, jsonify
import sqlite3

app = Flask(__name__)

# Database connection
def get_db_connection():
    conn = sqlite3.connect('db/attendance.db')
    conn.row_factory = sqlite3.Row  # This allows column access by name
    return conn

# Home route to show attendance capture (for simplicity, we're just showing a form for now)
@app.route('/')
def index():
    return render_template('index.html')

# Route to capture and mark attendance
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    student_name = request.form['student_name']
    emotion = request.form['emotion']
    
    # Insert into the database
    conn = get_db_connection()
    conn.execute('INSERT INTO attendance (student_name, emotion) VALUES (?, ?)', (student_name, emotion))
    conn.commit()
    conn.close()
    
    return jsonify({"message": "Attendance marked successfully!"})

# Report route to view attendance history
@app.route('/report')
def report():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance")
    records = cursor.fetchall()
    conn.close()
    return render_template('report.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
