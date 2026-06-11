"""
=============================================================================
STEP 1 — Audio Transcription using Whisper
Medical Billing Pipeline | Windows
=============================================================================
INSTALL (run once in your terminal):
    pip install faster-whisper

USAGE:
    # Single file:
    python step1_transcribe.py --file ENC0001.mp3

    # All 200 files in a folder:
    python step1_transcribe.py --folder C:/your/audio/folder

OUTPUT:
    transcripts/ENC0001.txt   → raw transcript text
    transcripts/ENC0001.json  → transcript + word-level timestamps
=============================================================================
"""

import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel


# =============================================================================
# CONFIG — change these if needed
# =============================================================================
MODEL_SIZE   = "small"    # tiny | base | small | medium  (small = best balance)
DEVICE       = "cpu"      # use "cuda" if you have an NVIDIA GPU
COMPUTE_TYPE = "int8"     # fastest on CPU
LANGUAGE     = "en"
OUTPUT_DIR   = "./transcripts"


# =============================================================================
# TRANSCRIBE ONE FILE
# =============================================================================
def transcribe_file(model, audio_path: str) -> dict:
    t0 = time.time()

    segments, info = model.transcribe(
        audio_path,
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,                          # skip silence automatically
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True
    )

    full_text = []
    segment_list = []

    for seg in segments:
        full_text.append(seg.text.strip())
        segment_list.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end, 2),
            "text":  seg.text.strip(),
            "words": [
                {
                    "word":        w.word.strip(),
                    "start":       round(w.start, 2),
                    "end":         round(w.end, 2),
                    "probability": round(w.probability, 4)
                }
                for w in (seg.words or [])
            ]
        })

    return {
        "file":           os.path.basename(audio_path),
        "transcript":     " ".join(full_text),
        "duration_sec":   round(info.duration, 2),
        "language":       info.language,
        "segments":       segment_list,
        "model":          MODEL_SIZE,
        "processing_sec": round(time.time() - t0, 2),
        "timestamp":      datetime.now().isoformat()
    }


# =============================================================================
# SAVE OUTPUTS
# =============================================================================
def save_outputs(result: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = Path(result["file"]).stem

    # Plain text
    txt_path = os.path.join(OUTPUT_DIR, f"{stem}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["transcript"])

    # Full JSON with metadata
    json_path = os.path.join(OUTPUT_DIR, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return txt_path, json_path


# =============================================================================
# SINGLE FILE
# =============================================================================
def run_single(model, audio_path: str):
    print(f"\n Transcribing: {audio_path}")
    result = transcribe_file(model, audio_path)
    txt_path, json_path = save_outputs(result)

    print(f"\n{'='*55}")
    print(f"  Done in {result['processing_sec']}s")
    print(f"  Audio duration : {result['duration_sec']}s")
    print(f"  Words extracted: {len(result['transcript'].split())}")
    print(f"\n  TRANSCRIPT PREVIEW:")
    print(f"  {result['transcript'][:400]}")
    print(f"\n  Saved: {txt_path}")
    print(f"  Saved: {json_path}")
    print(f"{'='*55}\n")

    return result


# =============================================================================
# BATCH — all files in a folder
# =============================================================================
def run_batch(model, folder_path: str):
    audio_files = sorted([
        f for f in Path(folder_path).iterdir()
        if f.suffix.lower() in (".mp3", ".wav", ".m4a", ".flac")
    ])

    if not audio_files:
        print(f"No audio files found in: {folder_path}")
        return

    print(f"\n Found {len(audio_files)} audio files")
    print(f" Model: {MODEL_SIZE} | Device: {DEVICE}\n")

    for i, audio_path in enumerate(audio_files, 1):
        # Skip if already transcribed
        txt_path = os.path.join(OUTPUT_DIR, f"{audio_path.stem}.txt")
        if os.path.exists(txt_path):
            print(f"[{i}/{len(audio_files)}] {audio_path.name} ... skipped (already done)")
            continue

        print(f"[{i}/{len(audio_files)}] {audio_path.name}", end=" ... ")
        try:
            result = transcribe_file(model, str(audio_path))
            save_outputs(result)
            print(f"done ({result['processing_sec']}s, {len(result['transcript'].split())} words)")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\n Batch complete. Transcripts saved to: {OUTPUT_DIR}/")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Whisper Medical Audio Transcription")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file",   type=str, help="Path to a single audio file")
    group.add_argument("--folder", type=str, help="Path to folder containing audio files")
    args = parser.parse_args()

    print(f"\n Loading Whisper '{MODEL_SIZE}' model ...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f" Model ready.\n")

    if args.file:
        run_single(model, args.file)
    else:
        run_batch(model, args.folder)


if __name__ == "__main__":
    main()
