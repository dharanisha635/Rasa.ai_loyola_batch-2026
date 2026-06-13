import sqlite3
import os
from datetime import datetime

DB_PATH = "agrostat.db"


def init_db():
    """Create tables if they don't exist. Call this once at app startup."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            plant_name TEXT,
            disease TEXT,
            confidence REAL,
            severity_level TEXT,
            severity_score REAL,
            risk_score REAL,
            affected_area REAL,
            spread_risk REAL,
            recovery_chance REAL,
            temp_min REAL,
            temp_max REAL,
            hum_min REAL,
            hum_max REAL,
            user_feedback TEXT DEFAULT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_scan(data: dict):
    """Save a scan result to the database. Call after every prediction."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO scans (
            timestamp, plant_name, disease, confidence,
            severity_level, severity_score, risk_score,
            affected_area, spread_risk, recovery_chance,
            temp_min, temp_max, hum_min, hum_max
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        data.get("plant_name"),
        data.get("disease"),
        data.get("confidence"),
        data.get("severity_level"),
        data.get("severity_score"),
        data.get("risk_score"),
        data.get("affected_area"),
        data.get("spread_risk"),
        data.get("recovery_chance"),
        data.get("temp_min"),
        data.get("temp_max"),
        data.get("hum_min"),
        data.get("hum_max")
    ))

    scan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return scan_id


def save_feedback(scan_id: int, feedback: str):
    """Save user feedback (correct/incorrect) for a scan."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE scans SET user_feedback = ? WHERE id = ?",
        (feedback, scan_id)
    )
    conn.commit()
    conn.close()


def get_all_scans():
    """Get all scans as a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans ORDER BY timestamp DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_disease_counts():
    """Get disease frequency counts for charts."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT disease, COUNT(*) as count
        FROM scans
        GROUP BY disease
        ORDER BY count DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [{"disease": r[0], "count": r[1]} for r in rows]


def get_severity_scores():
    """Get all severity scores for statistical analysis."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT severity_score FROM scans WHERE severity_score IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_feedback_labels():
    """Get scans that have user feedback for Precision/Recall/F1."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT disease, user_feedback
        FROM scans
        WHERE user_feedback IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_trend_data(days=30):
    """Get daily average severity for trend chart."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DATE(timestamp) as day, AVG(severity_score) as avg_sev
        FROM scans
        WHERE timestamp >= DATE('now', ?)
        GROUP BY DATE(timestamp)
        ORDER BY day ASC
    """, (f'-{days} days',))
    rows = cursor.fetchall()
    conn.close()
    return [{"day": r[0], "avg_severity": round(r[1], 2)} for r in rows]


def get_total_scans():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM scans")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def save_plant_correction(scan_id: int, correct_plant: str, image_hash: str):
    """Save user-corrected plant name linked to image hash."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plant_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER,
            correct_plant TEXT NOT NULL,
            image_hash TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        INSERT INTO plant_corrections (scan_id, correct_plant, image_hash, timestamp)
        VALUES (?, ?, ?, ?)
    """, (scan_id, correct_plant.strip(), image_hash, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()


def get_correction_by_hash(image_hash: str):
    """Check if this exact image was corrected before."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT correct_plant FROM plant_corrections
            WHERE image_hash = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (image_hash,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except:
        conn.close()
        return None


def get_most_corrected_plant(ai_plant: str):
    """
    If AI predicted a plant name that users frequently corrected,
    return the most common correction for it.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plant_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER,
                correct_plant TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            SELECT pc.correct_plant, COUNT(*) as cnt
            FROM plant_corrections pc
            JOIN scans s ON s.id = pc.scan_id
            WHERE LOWER(s.plant_name) = LOWER(?)
            GROUP BY pc.correct_plant
            ORDER BY cnt DESC
            LIMIT 1
        """, (ai_plant,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except:
        conn.close()
        return None


def get_all_corrections():
    """Get all corrections for analytics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plant_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER,
                correct_plant TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("SELECT * FROM plant_corrections ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows
    except:
        conn.close()
        return []