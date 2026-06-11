import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ──────────────────────────────────────────────────────
MODEL_PATH   = "model.keras"   # update if your filename differs
TEST_DIR     = "dataset/test"      # update to your actual test folder path
IMG_SIZE     = (128, 128)
BATCH_SIZE   = 1
MC_PASSES    = 50                  # Monte Carlo dropout passes
THRESHOLD    = 0.5
# ────────────────────────────────────────────────────────────────

model = load_model(MODEL_PATH)

test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

filenames   = test_data.filenames
true_labels = test_data.classes   # 0=Clean, 1=Dirty (alphabetical)

results = []

for i, (img_batch, true_label) in enumerate(test_data):
    if i >= len(filenames):
        break

    # Monte Carlo Dropout: run N passes with dropout ACTIVE
    mc_preds = np.array([
        model(img_batch, training=True)[0][0]  # training=True keeps dropout on
        for _ in range(MC_PASSES)
    ])

    mean_pred   = float(np.mean(mc_preds))
    uncertainty = float(np.std(mc_preds))
    label       = "Dirty" if mean_pred >= THRESHOLD else "Clean"
    confidence  = mean_pred if label == "Dirty" else (1 - mean_pred)
    is_ambiguous = uncertainty > 0.15

    results.append({
        "filename":     filenames[i],
        "true_label":   "Dirty" if true_labels[i] == 1 else "Clean",
        "predicted":    label,
        "confidence":   round(confidence, 4),
        "raw_score":    round(mean_pred, 4),
        "uncertainty":  round(uncertainty, 4),
        "is_ambiguous": is_ambiguous,
        "correct":      label == ("Dirty" if true_labels[i] == 1 else "Clean")
    })

    print(f"[{i+1}/{len(filenames)}] {filenames[i]} → {label} "
          f"(conf={confidence:.2f}, unc={uncertainty:.3f})")

# Save to JSON
with open("stats_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} predictions to stats_summary.json")