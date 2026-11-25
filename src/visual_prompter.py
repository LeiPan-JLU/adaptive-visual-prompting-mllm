# -*- coding: utf-8 -*-
"""Generate four visual-prompt variants for the CV-Bench subset.

Modes
-----
raw          original image
grid         semi-transparent 3x3 coordinate grid with A1-C3 labels
naive_som    fixed thick boxes + opaque black number tags (Set-of-Mark style)
adaptive_som density-aware rendering: thin boxes, compact tags, or centre dots

The density level is determined by the number of detected objects N and the
union box-area ratio rho.  Detection uses YOLOv8n; if it returns no boxes for
an icon-like image, a conservative contour fallback supplies proposals.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "data" / "eval_subset.jsonl"
CACHE = ROOT / "prompts_cache"
META = ROOT / "data" / "density_meta.jsonl"
PREVIEW = ROOT / "figures" / "preview_5_samples.png"
YOLO_WEIGHTS = ROOT / "models" / "yolov8n.pt"
MODES = ("raw", "grid", "naive_som", "adaptive_som")
COLORS = {
    "low": (70, 160, 80),
    "medium": (200, 150, 60),
    "high": (70, 90, 210),
}


def read_rows(limit: int | None = None):
    rows = [json.loads(line) for line in SUBSET.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def yolo_boxes(model, image: np.ndarray) -> list[tuple[int, int, int, int]]:
    result = model.predict(image, conf=0.25, device=0, verbose=False)[0]
    boxes = []
    if result.boxes is not None:
        for xyxy in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                boxes.append((x1, y1, x2, y2))
    return boxes


def contour_boxes(image: np.ndarray, max_boxes: int = 12) -> list[tuple[int, int, int, int]]:
    """Conservative fallback for non-COCO icon images with no YOLO detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    area_img = h * w
    boxes = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < area_img * 0.002 or area > area_img * 0.55:
            continue
        if bw < 8 or bh < 8:
            continue
        boxes.append((x, y, x + bw, y + bh))
        if len(boxes) >= max_boxes:
            break
    return boxes


def union_ratio(boxes, w: int, h: int) -> float:
    if not boxes:
        return 0.0
    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 255
    return float(mask.mean() / 255.0)


def density_level(n: int, rho: float) -> str:
    if n >= 9 or rho >= 0.60:
        return "High"
    if n >= 4 or rho >= 0.35:
        return "Medium"
    return "Low"


def put_text_alpha(canvas, text, org, scale, color, alpha, thickness=1):
    overlay = canvas.copy()
    cv2.putText(overlay, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(overlay, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def put_tag(canvas, text, x, y, scale, bg, fg, alpha=0.9):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(max(0, min(x, canvas.shape[1] - tw - 4)))
    y = int(max(th + 4, min(y, canvas.shape[0] - 2)))
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y - th - 4), (x + tw + 4, y + base), bg, -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    cv2.putText(canvas, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


def draw_grid(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    for i in range(1, 3):
        x = int(w * i / 3)
        y = int(h * i / 3)
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
    scale = max(0.45, min(0.7, min(h, w) / 650))
    for r in range(3):
        for c in range(3):
            label = f"{chr(ord('A') + r)}{c + 1}"
            put_tag(out, label, int(w * c / 3) + 4, int(h * r / 3) + 20, scale, (0, 0, 0), (255, 255, 255), alpha=0.35)
    return out


def draw_naive(image: np.ndarray, boxes) -> np.ndarray:
    out = image.copy()
    scale = max(0.45, min(0.9, min(out.shape[:2]) / 500 * 0.8))
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 40, 220), 3, cv2.LINE_AA)
        put_tag(out, str(i), x1 + 2, y1 + 2, scale, (0, 0, 0), (255, 255, 255), alpha=1.0)
    return out


def draw_adaptive(image: np.ndarray, boxes, level: str) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    short = min(h, w)
    color = COLORS[level.lower()]
    if not boxes:
        return out
    if level == "Low":
        width = max(1, int(round(short / 300)))
        scale = max(0.5, min(0.8, short / 550))
        for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, width, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
            put_tag(out, str(i), x1 + 2, y1 + 2, scale, color, (255, 255, 255), alpha=0.55)
    elif level == "Medium":
        scale = max(0.4, min(0.65, short / 750))
        for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
            put_tag(out, str(i), x1 + 1, y1 + 1, scale, color, (255, 255, 255), alpha=0.45)
    else:
        scale = max(0.38, min(0.6, short / 800))
        radius = max(2, int(round(short / 180)))
        for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            overlay = out.copy()
            cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
            put_tag(out, str(i), cx + radius + 1, cy - radius - 1, scale, (0, 0, 0), (255, 255, 255), alpha=0.45)
    return out


def preview(rows, out_path: Path):
    samples = rows[:5]
    cell_w, cell_h = 280, 210
    canvas = np.full((cell_h * len(samples) + 30, cell_w * 4, 3), 255, np.uint8)
    for r, row in enumerate(samples):
        for c, mode in enumerate(MODES):
            path = CACHE / mode / f"{row['id']}.jpg"
            img = load_image(path)
            scale = min((cell_w - 6) / img.shape[1], (cell_h - 24) / img.shape[0])
            resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
            y0 = r * cell_h + 24
            x0 = c * cell_w + (cell_w - resized.shape[1]) // 2
            canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
            cv2.putText(canvas, mode, (c * cell_w + 6, r * cell_h + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.limit)
    CACHE.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        (CACHE / mode).mkdir(parents=True, exist_ok=True)
    model = YOLO(str(YOLO_WEIGHTS)) if YOLO_WEIGHTS.exists() else YOLO("yolov8n.pt")
    meta = []
    for i, row in enumerate(rows, 1):
        img = load_image(ROOT / row["image_path"])
        boxes = yolo_boxes(model, img)
        source = "yolov8n"
        if not boxes:
            boxes = contour_boxes(img)
            source = "contour"
        boxes = boxes[:20]
        h, w = img.shape[:2]
        rho = union_ratio(boxes, w, h)
        level = density_level(len(boxes), rho)
        cv2.imwrite(str(CACHE / "raw" / f"{row['id']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(CACHE / "grid" / f"{row['id']}.jpg"), draw_grid(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(CACHE / "naive_som" / f"{row['id']}.jpg"), draw_naive(img, boxes), [cv2.IMWRITE_JPEG_QUALITY, 95])
        adaptive = draw_adaptive(img, boxes, level)
        cv2.imwrite(str(CACHE / "adaptive_som" / f"{row['id']}.jpg"), adaptive, [cv2.IMWRITE_JPEG_QUALITY, 95])
        meta.append({
            "id": row["id"],
            "task": row["task"],
            "num_boxes": len(boxes),
            "area_ratio": round(rho, 4),
            "density_level": level,
            "detector": source,
            "width": w,
            "height": h,
        })
        if i % 25 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] {row['id']} N={len(boxes)} rho={rho:.3f} {level}", flush=True)

    META.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in meta) + "\n", encoding="utf-8")
    if args.preview or args.limit:
        PREVIEW.parent.mkdir(parents=True, exist_ok=True)
        preview(rows, PREVIEW)
        print("preview", PREVIEW)
    levels = {k: sum(m["density_level"] == k for m in meta) for k in ("Low", "Medium", "High")}
    print("density levels", levels)


if __name__ == "__main__":
    main()
