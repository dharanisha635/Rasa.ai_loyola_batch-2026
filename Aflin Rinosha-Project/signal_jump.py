"""
signal_jump.py — Signal Jump Detection Module
==============================================
Integrates with the existing LPR Dashboard (v14).

Architecture
------------
TrafficLightMonitor  — reads a small ROI crop from the top-right of the frame
                       and uses HSV masking to decide RED / GREEN / UNKNOWN.

StopLineMonitor      — tracks whether a vehicle's bottom edge (y2 from the
                       YOLO bounding box) has crossed a fixed horizontal
                       pixel coordinate (the stop line).

SignalJumpDetector   — combines both: if the light is RED and a vehicle
                       crosses the stop line, it returns True (violation).

Usage inside app.py
-------------------
    from signal_jump import SignalJumpDetector

    _detector = SignalJumpDetector()          # one global instance

    # Inside the video frame loop:
    light = _detector.detect_light(frame)    # updates internal light state
    jumped = _detector.check_signal_jump(y2) # True if RED + crossed line

Configuration
-------------
    _detector.roi          = (x1, y1, x2, y2)  # pixel box of traffic light
    _detector.stop_line_y  = 400               # horizontal y pixel of stop line
    _detector.update_config({...})             # update from dict (POST body)
    _detector.get_config()                     # returns dict for GET response
"""

import cv2
import numpy as np
from typing import Tuple, Optional

# ─────────────────────────────────────────────────────────────
# HSV ranges for red and green traffic lights.
# Red wraps around the hue wheel so we need two ranges.
# Values tuned for typical street-level camera footage.
# ─────────────────────────────────────────────────────────────
SIGNAL_COLORS = {
    # Red: two hue bands (0-10 and 160-180)
    "red_lo1": (  0, 120,  80),
    "red_hi1": ( 10, 255, 255),
    "red_lo2": (160, 120,  80),
    "red_hi2": (180, 255, 255),
    # Green: 40-90 hue
    "green_lo": ( 40, 60,  60),
    "green_hi": ( 90, 255, 255),
}

# Minimum lit pixels to count as a detection (avoids noise)
MIN_LIT_PIXELS = 40


# ═════════════════════════════════════════════════════════════
# TRAFFIC LIGHT MONITOR
# ═════════════════════════════════════════════════════════════
class TrafficLightMonitor:
    """
    Detects the active colour of a circle-shaped traffic light
    by analysing HSV pixel density inside a configurable ROI.

    Parameters
    ----------
    roi : (x1, y1, x2, y2)
        Pixel bounding box of the traffic light in the full frame.
        Default is top-right corner, adjust to your camera layout.
    min_lit_pixels : int
        Minimum number of matching HSV pixels to declare a colour active.
    """

    def __init__(
        self,
        roi: Tuple[int, int, int, int] = (1100, 20, 1260, 180),
        min_lit_pixels: int = MIN_LIT_PIXELS,
    ):
        self.roi            = roi
        self.min_lit_pixels = min_lit_pixels
        self._last_color    = "unknown"

    # ── public ────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> str:
        """
        Analyse the ROI crop and return 'red', 'green', or 'unknown'.
        Result is also stored in self._last_color.
        """
        x1, y1, x2, y2 = self.roi
        fh, fw = frame.shape[:2]

        # Clamp ROI to frame dimensions
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(fw, x2); y2 = min(fh, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown"

        # Optional: apply a circular mask to isolate the lamp circle
        crop = self._apply_circle_mask(crop)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Red (two hue bands)
        mask_r1 = cv2.inRange(
            hsv,
            np.array(SIGNAL_COLORS["red_lo1"]),
            np.array(SIGNAL_COLORS["red_hi1"])
        )
        mask_r2 = cv2.inRange(
            hsv,
            np.array(SIGNAL_COLORS["red_lo2"]),
            np.array(SIGNAL_COLORS["red_hi2"])
        )
        red_px = cv2.countNonZero(mask_r1 | mask_r2)

        # Green
        mask_g  = cv2.inRange(
            hsv,
            np.array(SIGNAL_COLORS["green_lo"]),
            np.array(SIGNAL_COLORS["green_hi"])
        )
        green_px = cv2.countNonZero(mask_g)

        print(f"    [TL] red_px={red_px}  green_px={green_px}  min={self.min_lit_pixels}")

        if red_px >= self.min_lit_pixels and red_px >= green_px:
            self._last_color = "red"
        elif green_px >= self.min_lit_pixels:
            self._last_color = "green"
        else:
            # No strong colour — keep last known state (avoids flicker)
            pass

        return self._last_color

    @property
    def last_color(self) -> str:
        return self._last_color

    # ── private ───────────────────────────────────────────────
    @staticmethod
    def _apply_circle_mask(crop: np.ndarray) -> np.ndarray:
        """
        Creates an elliptical mask centred on the crop so that
        background colour outside the lamp circle doesn't pollute the count.
        """
        h, w = crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        rx, ry = max(1, w // 2 - 4), max(1, h // 2 - 4)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        result = cv2.bitwise_and(crop, crop, mask=mask)
        return result


# ═════════════════════════════════════════════════════════════
# STOP LINE MONITOR
# ═════════════════════════════════════════════════════════════
class StopLineMonitor:
    """
    Detects whether a vehicle has crossed a fixed horizontal stop line.

    The stop line is defined by a single Y pixel coordinate.
    A vehicle "crosses" when its bounding-box bottom edge (y2)
    moves from ABOVE the line (y2 < stop_line_y) to
    AT or BELOW it (y2 >= stop_line_y).

    Parameters
    ----------
    stop_line_y : int
        Y pixel coordinate of the stop line in the full frame.
        Vehicles approach from the top (smaller y) toward the line.
    tolerance : int
        Extra pixels of tolerance below the line to reduce false positives
        on slightly mis-sized bounding boxes. Default 10 px.
    """

    def __init__(self, stop_line_y: int = 400, tolerance: int = 10):
        self.stop_line_y = stop_line_y
        self.tolerance   = tolerance

    def has_crossed(self, vehicle_y2: int) -> bool:
        """
        Returns True if the bottom of the vehicle bounding box
        is at or past the stop line (within tolerance).
        """
        return vehicle_y2 >= (self.stop_line_y - self.tolerance)


# ═════════════════════════════════════════════════════════════
# SIGNAL JUMP DETECTOR  (combines both monitors)
# ═════════════════════════════════════════════════════════════
class SignalJumpDetector:
    """
    High-level API used by app.py.

    Workflow per video frame:
        1. Call detect_light(frame) — updates internal light state.
        2. For each detected vehicle call check_signal_jump(y2).
           Returns True only when:
               • light is RED   AND
               • vehicle y2 >= stop_line_y

    Works for car, bike, and truck equally — the caller decides vtype.

    Configuration can be changed at runtime via update_config() / get_config()
    so the frontend can POST new ROI / stop-line values without restarting Flask.
    """

    def __init__(
        self,
        roi:          Tuple[int, int, int, int] = (1100, 20, 1260, 180),
        stop_line_y:  int  = 400,
        tolerance:    int  = 10,
        min_lit_pixels: int = MIN_LIT_PIXELS,
        enabled:      bool = True,
    ):
        self._tl  = TrafficLightMonitor(roi=roi, min_lit_pixels=min_lit_pixels)
        self._sl  = StopLineMonitor(stop_line_y=stop_line_y, tolerance=tolerance)
        self.enabled = enabled

    # ── properties that proxy to inner objects ─────────────────
    @property
    def roi(self) -> Tuple[int, int, int, int]:
        return self._tl.roi

    @roi.setter
    def roi(self, value):
        self._tl.roi = tuple(value)

    @property
    def stop_line_y(self) -> int:
        return self._sl.stop_line_y

    @stop_line_y.setter
    def stop_line_y(self, value: int):
        self._sl.stop_line_y = int(value)

    # ── public API ─────────────────────────────────────────────
    def detect_light(self, frame: np.ndarray) -> str:
        """
        Read the traffic light ROI and update internal state.
        Returns 'red', 'green', or 'unknown'.
        Call once per frame before processing vehicles.
        """
        if not self.enabled:
            return "unknown"
        return self._tl.detect(frame)

    def check_signal_jump(self, vehicle_y2: int) -> bool:
        """
        Returns True if:
          - Detection is enabled
          - Current light is RED
          - vehicle_y2 has crossed the stop line

        Parameters
        ----------
        vehicle_y2 : int
            Bottom pixel coordinate (y2) of the vehicle YOLO bounding box.
        """
        if not self.enabled:
            return False
        if self._tl.last_color != "red":
            return False
        return self._sl.has_crossed(vehicle_y2)

    def get_config(self) -> dict:
        """Serialise current config for GET /signal_config."""
        return {
            "enabled":       self.enabled,
            "roi":           list(self._tl.roi),
            "stop_line_y":   self._sl.stop_line_y,
            "tolerance":     self._sl.tolerance,
            "min_lit_pixels": self._tl.min_lit_pixels,
            "current_light": self._tl.last_color,
        }

    def update_config(self, data: dict):
        """
        Update config from a dict (POST /signal_config body).
        Only keys that are present are updated.
        """
        if "enabled" in data:
            self.enabled = bool(data["enabled"])
        if "roi" in data:
            roi = data["roi"]
            if len(roi) != 4:
                raise ValueError("roi must be [x1, y1, x2, y2]")
            self._tl.roi = tuple(int(v) for v in roi)
        if "stop_line_y" in data:
            self._sl.stop_line_y = int(data["stop_line_y"])
        if "tolerance" in data:
            self._sl.tolerance = int(data["tolerance"])
        if "min_lit_pixels" in data:
            self._tl.min_lit_pixels = int(data["min_lit_pixels"])