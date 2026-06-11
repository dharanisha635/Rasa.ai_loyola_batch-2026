"""
=============================================================================
STEP 3 — Save Extracted Records to MySQL
Medical Billing Pipeline | Windows
=============================================================================
INSTALL (run once):
    pip install mysql-connector-python

MYSQL SETUP (run in MySQL Workbench or command line):
    CREATE DATABASE medical_billing;

USAGE:
    # Insert all extracted records from all_records.json:
    python step3_database.py --file extracted/all_records.json

    # Insert a single extracted record:
    python step3_database.py --file extracted/ENC0001_extracted.json
=============================================================================
"""

import json
import argparse
from pathlib import Path
import mysql.connector
from mysql.connector import Error


# =============================================================================
# CONFIG — update with your MySQL credentials
# =============================================================================
from dotenv import load_dotenv
import os
load_dotenv()


DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "port":     3306,
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

# =============================================================================
# CREATE TABLE (runs automatically on first use)
# =============================================================================
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS encounters (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    encounter_id     VARCHAR(20)    UNIQUE NOT NULL,
    encounter_date   DATE,
    dictation_date   DATE,
    patient_age      INT,
    patient_gender   CHAR(1),
    department       VARCHAR(50),
    diagnosis        VARCHAR(200),
    icd10_code       VARCHAR(20),
    cpt_code         VARCHAR(20),
    billed_amount    DECIMAL(10,2),
    claim_status     VARCHAR(20),
    is_rushed        BOOLEAN,
    notes            TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

INSERT_SQL = """
INSERT INTO encounters (
    encounter_id, encounter_date, dictation_date,
    patient_age, patient_gender, department,
    diagnosis, icd10_code, cpt_code,
    billed_amount, claim_status, is_rushed, notes
) VALUES (
    %(encounter_id)s, %(encounter_date)s, %(dictation_date)s,
    %(patient_age)s, %(patient_gender)s, %(department)s,
    %(diagnosis)s, %(icd10_code)s, %(cpt_code)s,
    %(billed_amount)s, %(claim_status)s, %(is_rushed)s, %(notes)s
)
ON DUPLICATE KEY UPDATE
    encounter_date  = VALUES(encounter_date),
    dictation_date  = VALUES(dictation_date),
    patient_age     = VALUES(patient_age),
    patient_gender  = VALUES(patient_gender),
    department      = VALUES(department),
    diagnosis       = VALUES(diagnosis),
    icd10_code      = VALUES(icd10_code),
    cpt_code        = VALUES(cpt_code),
    billed_amount   = VALUES(billed_amount),
    claim_status    = VALUES(claim_status),
    is_rushed       = VALUES(is_rushed),
    notes           = VALUES(notes);
"""


# =============================================================================
# DB CONNECTION
# =============================================================================
def get_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"\n  MySQL connection failed: {e}")
        print(f"  Check your DB_CONFIG credentials in step3_database.py")
        raise


# =============================================================================
# ENSURE TABLE EXISTS
# =============================================================================
def ensure_table(conn):
    cursor = conn.cursor()
    cursor.execute(CREATE_TABLE_SQL)
    conn.commit()
    cursor.close()
    print("  Table 'encounters' ready.")


# =============================================================================
# CLEAN ONE RECORD BEFORE INSERT
# Handles type conversions (bool, None, date strings)
# =============================================================================
def clean_record(record: dict) -> dict:
    cleaned = {}

    for key, value in record.items():
        # Skip any extra keys not in our schema
        if key not in [
            "encounter_id", "encounter_date", "dictation_date",
            "patient_age", "patient_gender", "department",
            "diagnosis", "icd10_code", "cpt_code",
            "billed_amount", "claim_status", "is_rushed", "notes"
        ]:
            continue

        # Convert empty strings to None
        if value == "" or value == "null":
            cleaned[key] = None
        # Convert is_rushed to proper boolean
        elif key == "is_rushed":
            if isinstance(value, bool):
                cleaned[key] = value
            elif isinstance(value, str):
                cleaned[key] = value.lower() in ("true", "yes", "1")
            else:
                cleaned[key] = None
        # Convert billed_amount to float
        elif key == "billed_amount" and value is not None:
            try:
                cleaned[key] = float(str(value).replace("$", "").replace(",", ""))
            except (ValueError, TypeError):
                cleaned[key] = None
        # Convert patient_age to int
        elif key == "patient_age" and value is not None:
            try:
                cleaned[key] = int(value)
            except (ValueError, TypeError):
                cleaned[key] = None
        else:
            cleaned[key] = value

    # Fill any missing keys with None
    schema_keys = [
        "encounter_id", "encounter_date", "dictation_date",
        "patient_age", "patient_gender", "department",
        "diagnosis", "icd10_code", "cpt_code",
        "billed_amount", "claim_status", "is_rushed", "notes"
    ]
    for k in schema_keys:
        if k not in cleaned:
            cleaned[k] = None

    return cleaned


# =============================================================================
# INSERT ONE RECORD
# =============================================================================
def insert_record(cursor, record: dict) -> bool:
    cleaned = clean_record(record)
    try:
        cursor.execute(INSERT_SQL, cleaned)
        return True
    except Error as e:
        print(f"    Insert error for {record.get('encounter_id', '?')}: {e}")
        return False


# =============================================================================
# MAIN LOAD FUNCTION
# =============================================================================
def load_records(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both a single record (dict) and a list of records
    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        print("Unexpected JSON format.")
        return

    print(f"\n  Loaded {len(records)} record(s) from {json_path}")

    conn = get_connection()
    ensure_table(conn)

    cursor = conn.cursor()
    inserted = 0
    failed   = 0

    for i, record in enumerate(records, 1):
        eid = record.get("encounter_id", f"record_{i}")
        success = insert_record(cursor, record)
        if success:
            inserted += 1
            print(f"  [{i}/{len(records)}] {eid} — inserted/updated")
        else:
            failed += 1

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\n{'='*55}")
    print(f"  Done. {inserted} inserted/updated, {failed} failed.")
    print(f"  Database: {DB_CONFIG['database']} @ {DB_CONFIG['host']}")
    print(f"{'='*55}\n")


# =============================================================================
# VERIFY — print first 5 rows from DB
# =============================================================================
def verify_data():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT encounter_id, department, icd10_code, billed_amount, claim_status FROM encounters LIMIT 5;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    print("\n  Sample rows from database:")
    print(f"  {'encounter_id':<12} {'department':<18} {'icd10':<10} {'amount':>8}  {'status'}")
    print(f"  {'-'*65}")
    for row in rows:
        print(f"  {row['encounter_id']:<12} {str(row['department']):<18} "
              f"{str(row['icd10_code']):<10} "
              f"${str(row['billed_amount']):>7}  {row['claim_status']}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    if DB_CONFIG["password"] == "YOUR_PASSWORD":
        print("ERROR: Please set your MySQL password in the DB_CONFIG section.")
        return

    parser = argparse.ArgumentParser(description="Load extracted records into MySQL")
    parser.add_argument("--file",   required=True, help="Path to JSON file (single or all_records.json)")
    parser.add_argument("--verify", action="store_true", help="Print sample rows after loading")
    args = parser.parse_args()

    load_records(args.file)

    if args.verify:
        verify_data()


if __name__ == "__main__":
    main()
