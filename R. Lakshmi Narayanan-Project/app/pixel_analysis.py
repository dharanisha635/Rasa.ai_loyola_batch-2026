import cv2
import numpy as np


def measure_affected_area(image_path: str) -> dict:
    """
    Uses OpenCV to measure the real percentage of diseased
    vs healthy tissue based on pixel colour analysis.
    Returns a dict with affected_area_real and healthy_area_real.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"affected_area_real": None, "healthy_area_real": None}

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Diseased regions: brown, yellow, dark spots
        masks = []

        # Brown / tan diseased tissue
        masks.append(cv2.inRange(hsv,
            np.array([10, 40, 40]),
            np.array([25, 255, 200])
        ))

        # Yellow diseased tissue
        masks.append(cv2.inRange(hsv,
            np.array([25, 40, 100]),
            np.array([35, 255, 255])
        ))

        # Dark necrotic spots (very dark pixels)
        masks.append(cv2.inRange(hsv,
            np.array([0, 0, 0]),
            np.array([180, 255, 50])
        ))

        # Combine all disease masks
        disease_mask = masks[0]
        for m in masks[1:]:
            disease_mask = cv2.bitwise_or(disease_mask, m)

        # Healthy green tissue
        healthy_mask = cv2.inRange(hsv,
            np.array([35, 40, 40]),
            np.array([85, 255, 255])
        )

        disease_pixels = int(cv2.countNonZero(disease_mask))
        healthy_pixels = int(cv2.countNonZero(healthy_mask))
        total = disease_pixels + healthy_pixels

        if total == 0:
            return {"affected_area_real": 0.0, "healthy_area_real": 100.0}

        affected_pct = round((disease_pixels / total) * 100, 1)
        healthy_pct  = round((healthy_pixels / total) * 100, 1)

        return {
            "affected_area_real": min(affected_pct, 100.0),
            "healthy_area_real":  min(healthy_pct,  100.0)
        }

    except Exception as e:
        print(f"Pixel analysis error: {e}")
        return {"affected_area_real": None, "healthy_area_real": None}