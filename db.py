import sqlite3
import json
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE_NAME = "job_portal.db"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    """Creates the necessary tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('student', 'tnp'))
    )
    """)

    # Jobs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tnp_id INTEGER NOT NULL,
        job_title TEXT NOT NULL,
        jd_text TEXT NOT NULL,
        jd_data_json TEXT NOT NULL,
        FOREIGN KEY (tnp_id) REFERENCES users (id)
    )
    """)

    # Applications table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        job_id INTEGER NOT NULL,
        resume_text TEXT NOT NULL,
        resume_data_json TEXT NOT NULL,
        relevance_score REAL NOT NULL,
        FOREIGN KEY (student_id) REFERENCES users (id),
        FOREIGN KEY (job_id) REFERENCES jobs (id),
        UNIQUE(student_id, job_id)
    )
    """)

    conn.commit()
    conn.close()

# --- User Management ---
def add_user(username, password, role):
    """Adds a new user to the database."""
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), role),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Username already exists
    finally:
        conn.close()

def get_user(username):
    """Retrieves a user by their username."""
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return user

def check_password(password_hash, password):
    """Verifies a password against its hash."""
    return check_password_hash(password_hash, password)


# --- Job Management (TnP) ---
def add_job(tnp_id, job_title, jd_text, jd_data):
    """Adds a new job posting to the database."""
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO jobs (tnp_id, job_title, jd_text, jd_data_json) VALUES (?, ?, ?, ?)",
        (tnp_id, job_title, jd_text, json.dumps(jd_data)),
    )
    conn.commit()
    conn.close()

def get_tnp_jobs(tnp_id):
    """Gets all jobs posted by a specific TnP user."""
    conn = get_db_connection()
    jobs = conn.execute("SELECT * FROM jobs WHERE tnp_id = ? ORDER BY id DESC", (tnp_id,)).fetchall()
    conn.close()
    return jobs

# --- Application Management ---
def add_application(student_id, job_id, resume_text, resume_data, score):
    """Adds a student's application for a job."""
    conn = get_db_connection()
    try:
        conn.execute(
            """INSERT INTO applications (student_id, job_id, resume_text, resume_data_json, relevance_score)
               VALUES (?, ?, ?, ?, ?)""",
            (student_id, job_id, resume_text, json.dumps(resume_data), score)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError: # student has already applied for this job
        return False
    finally:
        conn.close()

def get_applications_for_job(job_id):
    """Retrieves all applications for a specific job, including student info."""
    conn = get_db_connection()
    query = """
    SELECT a.*, u.username
    FROM applications a
    JOIN users u ON a.student_id = u.id
    WHERE a.job_id = ?
    ORDER BY a.relevance_score DESC
    """
    applications = conn.execute(query, (job_id,)).fetchall()
    conn.close()
    return applications

def get_all_jobs():
    """Gets all available jobs for students to view."""
    conn = get_db_connection()
    jobs = conn.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
    conn.close()
    return jobs
    
def get_student_applications(student_id):
    """Gets all applications submitted by a specific student."""
    conn = get_db_connection()
    query = """
        SELECT a.relevance_score, j.job_title
        FROM applications a
        JOIN jobs j ON a.job_id = j.id
        WHERE a.student_id = ?
        ORDER BY j.id DESC
    """
    apps = conn.execute(query, (student_id,)).fetchall()
    conn.close()
    return apps

# Initialize the database on first run
create_tables()
