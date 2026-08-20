"""
YoloTrackX - Helper utilities for detection, tracking and visualization.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Color palette (burgundy / gold / cream theme)
# ---------------------------------------------------------------------------
BURGUNDY = (45, 25, 128)       # BGR
GOLD = (0, 180, 220)           # BGR
CREAM = (230, 240, 255)        # BGR
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)

TRACK_COLORS = [
    (45, 25, 128),
    (0, 180, 220),
    (80, 60, 200),
    (30, 140, 180),
    (100, 40, 160),
    (50, 200, 240),
    (70, 50, 140),
    (20, 160, 200),
]


def get_track_color(track_id: Optional[int]) -> Tuple[int, int, int]:
    if track_id is None:
        return BURGUNDY
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]


def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    return YOLO(model_name)


def get_available_models() -> Dict[str, str]:
    return {
        "YOLOv8 Nano (fastest)": "yolov8n.pt",
        "YOLOv8 Small": "yolov8s.pt",
        "YOLOv8 Medium": "yolov8m.pt",
        "YOLOv8 Large": "yolov8l.pt",
        "YOLOv8 XLarge (most accurate)": "yolov8x.pt",
    }


def draw_detections(
    frame: np.ndarray,
    result,
    show_labels: bool = True,
    show_conf: bool = True,
    show_track_id: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    annotated = frame.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    track_ids = None
    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)

    names = result.names

    for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confs)):
        x1, y1, x2, y2 = map(int, box)
        tid = int(track_ids[i]) if track_ids is not None else None
        color = get_track_color(tid if tid is not None else cls_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        parts = []
        if show_labels:
            parts.append(names.get(cls_id, str(cls_id)))
        if show_conf:
            parts.append(f"{conf:.2f}")
        if show_track_id and tid is not None:
            parts.append(f"ID:{tid}")

        label = " | ".join(parts)
        if not label:
            continue

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - th - baseline - 6),
            (x1 + tw + 6, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            WHITE,
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_stats_overlay(
    frame: np.ndarray,
    fps: float,
    total_objects: int,
    class_counts: Dict[str, int],
    unique_tracks: int = 0,
) -> np.ndarray:
    overlay = frame.copy()
    h, w = frame.shape[:2]

    panel_w = min(280, w // 3)
    panel_h = 90 + len(class_counts) * 22
    panel_h = min(panel_h, h // 2)

    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 30
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (18, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GOLD, 2, cv2.LINE_AA
    )
    y += 25
    cv2.putText(
        frame, f"Objects: {total_objects}", (18, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CREAM, 1, cv2.LINE_AA
    )
    y += 22
    if unique_tracks > 0:
        cv2.putText(
            frame, f"Unique tracks: {unique_tracks}", (18, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, CREAM, 1, cv2.LINE_AA
        )
        y += 22

    for cls_name, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        if y > 8 + panel_h - 10:
            break
        cv2.putText(
            frame, f"{cls_name}: {cnt}", (18, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA
        )
        y += 20

    return frame


def count_classes(result) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    if result.boxes is None or len(result.boxes) == 0:
        return counts
    classes = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names
    for c in classes:
        counts[names.get(int(c), str(c))] += 1
    return dict(counts)


def extract_detections_table(result) -> List[Dict]:
    rows = []
    if result.boxes is None or len(result.boxes) == 0:
        return rows

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    track_ids = None
    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)

    names = result.names
    for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confs)):
        x1, y1, x2, y2 = box
        row = {
            "class": names.get(int(cls_id), str(cls_id)),
            "confidence": round(float(conf), 4),
            "x1": round(float(x1), 1),
            "y1": round(float(y1), 1),
            "x2": round(float(x2), 1),
            "y2": round(float(y2), 1),
            "track_id": int(track_ids[i]) if track_ids is not None else None,
        }
        rows.append(row)
    return rows


def resize_frame(frame: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)