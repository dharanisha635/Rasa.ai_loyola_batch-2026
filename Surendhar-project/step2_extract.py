"""
=============================================================================
STEP 2 — Structured Field Extraction using Groq (Key Rotation)
Medical Billing Pipeline | Windows
=============================================================================
INSTALL (run once):
    pip install groq python-dotenv

.env FILE SETUP:
    GROQ_API_KEY_1=your_first_key
    GROQ_API_KEY_2=your_second_key
    GROQ_API_KEY_3=your_third_key

USAGE:
    # Single transcript:
    python step2_extract.py --file transcripts/ENC0001.txt

    # All transcripts:
    python step2_extract.py --folder transcripts/

OUTPUT:
    extracted/ENC0001_extracted.json
    extracted/all_records.json
=============================================================================
"""

import os
import json
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# =============================================================================
# CONFIG
# =============================================================================
load_dotenv()
OUTPUT_DIR = "./extracted"
MODEL_NAME = "llama-3.3-70b-versatile"

# =============================================================================
# KEY ROTATION SETUP
# =============================================================================
API_KEYS = []
for i in range(1, 4):
    key = os.environ.get(f"GROQ_API_KEY_{i}")
    if key:
        API_KEYS.append(key)

if not API_KEYS:
    single_key = os.environ.get("GROQ_API_KEY")
    if single_key:
        API_KEYS.append(single_key)

print(f"  Loaded {len(API_KEYS)} Groq API key(s) for rotation.")

current_key_index = 0

def get_client():
    """Returns a Groq client using the current key."""
    return Groq(api_key=API_KEYS[current_key_index])

def rotate_key():
    """Rotates to the next available API key."""
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    print(f"    Rotated to API key {current_key_index + 1}")

# =============================================================================
# DEPARTMENT FALLBACK MAPS
# =============================================================================
DIAGNOSIS_DEPARTMENT_MAP = {
    "ankle sprain":    "Orthopedics",
    "radius fracture": "Orthopedics",
    "fracture":        "Orthopedics",
    "sprain":          "Orthopedics",
    "strapping":       "Orthopedics",
    "bronchitis":      "General Medicine",
    "strep throat":    "General Medicine",
    "strep":           "General Medicine",
    "pharyngitis":     "General Medicine",
    "abscess":         "Dermatology",
    "laceration":      "Dermatology",
    "skin":            "Dermatology",
    "wound":           "Dermatology",
}

ICD_DEPARTMENT_MAP = {
    # Orthopedics
    "S93": "Orthopedics",
    "S52": "Orthopedics",
    # General Medicine
    "J20": "General Medicine",
    "J02": "General Medicine",
    # Dermatology
    "L02": "Dermatology",
    "S61": "Dermatology",
    # Cardiology — ADD THESE
    "R07": "Cardiology",
    "I10": "Cardiology",
    # Pediatrics — ADD THESE
    "H66": "Pediatrics",
    "R50": "Pediatrics",
    # Neurology — ADD THESE
    "G43": "Neurology",
    "H81": "Neurology",
}

# =============================================================================
# EXTRACTION PROMPT
# =============================================================================
EXTRACTION_PROMPT = """
You are a medical billing data extraction engine.
Read the clinical dictation transcript below and extract ALL fields.
Return ONLY a valid JSON object — no explanation, no markdown, no extra text.

STRICT RULES:
1. encounter_id   : use the filename provided (e.g. ENC0001)
2. encounter_date : look for "Encounter date" in transcript → YYYY-MM-DD format
3. dictation_date : use same date as encounter_date if no separate date mentioned
4. patient_age    : integer, look for "year-old"
5. patient_gender : "M" or "F", look for male/female
6. department     : MUST infer from diagnosis:
                    - ankle sprain, fracture, strapping, taping → "Orthopedics"
                    - bronchitis, strep throat, pharyngitis → "General Medicine"
                    - abscess, laceration, skin wound → "Dermatology"
                    - chest pain, hypertension → "Cardiology"
                    - ear infection, otitis, fever → "Pediatrics"
                    - migraine, vertigo → "Neurology"
7. diagnosis      : plain English diagnosis name
8. icd10_code     : look for "ICD-10 code" followed by the code
9. cpt_code       : look for "CPT code" followed by the code
10. billed_amount : look for "billed amount" or "$" followed by number → float
11. claim_status  : ALWAYS use "Pending" if not explicitly mentioned
12. is_rushed     : true if transcript says "rushed", false otherwise. NEVER null.
13. notes         : any remaining clinical observations as a string

Filename: {encounter_id}

Transcript:
{transcript}
"""

# =============================================================================
# PYTHON-LEVEL FALLBACKS
# =============================================================================
def apply_fallbacks(data: dict, transcript: str) -> dict:

    # Fix department
    if not data.get("department"):
        diagnosis        = (data.get("diagnosis") or "").lower()
        icd              = (data.get("icd10_code") or "")
        transcript_lower = transcript.lower()

        for keyword, dept in DIAGNOSIS_DEPARTMENT_MAP.items():
            if keyword in diagnosis:
                data["department"] = dept
                break

        if not data.get("department"):
            for prefix, dept in ICD_DEPARTMENT_MAP.items():
                if icd.startswith(prefix):
                    data["department"] = dept
                    break

        if not data.get("department"):
            for keyword, dept in DIAGNOSIS_DEPARTMENT_MAP.items():
                if keyword in transcript_lower:
                    data["department"] = dept
                    break

    # Fix dictation_date
    if not data.get("dictation_date") and data.get("encounter_date"):
        data["dictation_date"] = data["encounter_date"]

    # Fix claim_status
    if not data.get("claim_status"):
        data["claim_status"] = "Pending"

    # Fix is_rushed
    if data.get("is_rushed") is None:
        data["is_rushed"] = "rushed" in transcript.lower()

    return data


# =============================================================================
# EXTRACT ONE TRANSCRIPT
# =============================================================================
def extract_fields(transcript_text: str, encounter_id: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(
        encounter_id=encounter_id,
        transcript=transcript_text
    )

    for attempt in range(len(API_KEYS) * 3):  # try all keys multiple times
        try:
            client   = get_client()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if model adds them
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw   = "\n".join(lines).strip()

            data = json.loads(raw)
            data["encounter_id"] = encounter_id
            data = apply_fallbacks(data, transcript_text)
            return data

        except json.JSONDecodeError as e:
            print(f"    JSON parse error (attempt {attempt+1}): {e}")
            time.sleep(2)

        except Exception as e:
            error_msg = str(e)
            print(f"    Groq error (attempt {attempt+1}): {error_msg[:80]}")

            # Rate limit — rotate key and wait
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                rotate_key()
                time.sleep(5)
            else:
                time.sleep(3)

    # All attempts failed — apply fallbacks on empty skeleton
    print(f"    WARNING: All keys failed for {encounter_id}, using fallback only.")
    skeleton = {
        "encounter_id":   encounter_id,
        "encounter_date": None,
        "dictation_date": None,
        "patient_age":    None,
        "patient_gender": None,
        "department":     None,
        "diagnosis":      None,
        "icd10_code":     None,
        "cpt_code":       None,
        "billed_amount":  None,
        "claim_status":   None,
        "is_rushed":      None,
        "notes":          None
    }
    return apply_fallbacks(skeleton, transcript_text)


# =============================================================================
# SAVE OUTPUTS
# =============================================================================
def save_single(data: dict, encounter_id: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{encounter_id}_extracted.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_path


def save_all_records(all_records: list):
    out_path = os.path.join(OUTPUT_DIR, "all_records.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n  Combined records saved: {out_path}")


# =============================================================================
# SINGLE FILE
# =============================================================================
def run_single(txt_path: str):
    encounter_id = Path(txt_path).stem

    with open(txt_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    print(f"\n Extracting: {txt_path}")
    data     = extract_fields(transcript, encounter_id)
    out_path = save_single(data, encounter_id)

    print(f"\n{'='*55}")
    for k, v in data.items():
        print(f"  {k:<20} : {v}")
    print(f"\n  Saved: {out_path}")
    print(f"{'='*55}\n")
    return data


# =============================================================================
# BATCH
# =============================================================================
def run_batch(folder_path: str):
    txt_files = sorted([
        f for f in Path(folder_path).iterdir()
        if f.suffix.lower() == ".txt"
    ])

    if not txt_files:
        print(f"No .txt files found in: {folder_path}")
        return

    print(f"\n Found {len(txt_files)} transcript files\n")
    all_records = []

    for i, txt_path in enumerate(txt_files, 1):
        encounter_id = txt_path.stem

        # Skip if already extracted
        out_path = os.path.join(OUTPUT_DIR, f"{encounter_id}_extracted.json")
        if os.path.exists(out_path):
            print(f"[{i}/{len(txt_files)}] {txt_path.name} ... skipped (already done)")
            continue

        print(f"[{i}/{len(txt_files)}] {txt_path.name}", end=" ... ")

        with open(txt_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        data = extract_fields(transcript, encounter_id)
        save_single(data, encounter_id)
        all_records.append(data)

        print(f"done — {data.get('department','?')} | "
              f"{data.get('icd10_code','?')} | "
              f"${data.get('billed_amount','?')} | "
              f"rushed={data.get('is_rushed','?')}")

        time.sleep(1)  # small delay between requests

    # Load all records including skipped ones for summary
    all_json_files = sorted(Path(OUTPUT_DIR).glob("*_extracted.json"))
    all_records = []
    for jf in all_json_files:
        with open(jf, "r", encoding="utf-8") as f:
            all_records.append(json.load(f))

    save_all_records(all_records)
    print(f"\n Batch complete. {len(all_records)} total records.")

    # Null summary
    print("\n NULL VALUE SUMMARY:")
    for field in ["department", "dictation_date", "claim_status", "is_rushed", "diagnosis", "billed_amount"]:
        null_count = sum(1 for r in all_records if r.get(field) is None)
        status     = "OK" if null_count == 0 else "NEEDS ATTENTION"
        print(f"  {field:<20} : {null_count:>3} nulls — {status}")

    return all_records


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not API_KEYS:
        print("ERROR: No Groq API keys found. Check your .env file.")
        print("Expected: GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3")
        return

    parser = argparse.ArgumentParser(description="Groq Medical Billing Extractor")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file",   type=str, help="Path to a single .txt transcript")
    group.add_argument("--folder", type=str, help="Path to folder of .txt transcripts")
    args = parser.parse_args()

    if args.file:
        run_single(args.file)
    else:
        run_batch(args.folder)


if __name__ == "__main__":
    main()
