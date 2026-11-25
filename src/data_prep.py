# -*- coding: utf-8 -*-
"""Build a fixed 400-question CV-Bench 2D subset for the AVP experiments.

The source parquet is downloaded from ModelScope:
    https://modelscope.cn/datasets/comefly/cvbench

Outputs:
    data/eval_subset.jsonl
    data/images/{id}.jpg
"""
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "test_2d.parquet"
IMAGES = ROOT / "data" / "images"
SUBSET = ROOT / "data" / "eval_subset.jsonl"
SEED = 42
PER_TASK = 200


def answer_letter(raw) -> str:
    text = str(raw).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return text.strip().upper()[:1]


def extract_image(cell) -> Image.Image:
    if isinstance(cell, dict):
        data = cell.get("bytes")
        if data is None and cell.get("path"):
            data = Path(cell["path"]).read_bytes()
        if data is None:
            raise ValueError("image cell has neither bytes nor path")
        return Image.open(io.BytesIO(data)).convert("RGB")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    if isinstance(cell, str) and Path(cell).exists():
        return Image.open(cell).convert("RGB")
    raise TypeError(f"unsupported image cell: {type(cell)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task", type=int, default=PER_TASK)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not RAW.exists():
        raise FileNotFoundError(f"missing {RAW}; download the CV-Bench 2D parquet first")
    df = pd.read_parquet(RAW)
    IMAGES.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    for task in ("Count", "Relation"):
        pool = df[df["task"] == task]
        n = min(args.per_task, len(pool))
        picked = pool.sample(n=n, random_state=args.seed).to_dict("records")
        rng.shuffle(picked)
        for i, row in enumerate(picked):
            item_id = f"{'C' if task == 'Count' else 'R'}{i:04d}"
            image = extract_image(row["image"])
            image.save(IMAGES / f"{item_id}.jpg", quality=95)
            choices = [str(x) for x in list(row["choices"])]
            rows.append({
                "id": item_id,
                "task": task,
                "question": str(row["question"]),
                "prompt": str(row["prompt"]),
                "choices": choices,
                "answer": answer_letter(row["answer"]),
                "image_path": f"data/images/{item_id}.jpg",
                "source": str(row["source"]),
                "source_filename": str(row["source_filename"]),
            })

    IMAGES.mkdir(parents=True, exist_ok=True)
    with SUBSET.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {task: sum(r["task"] == task for r in rows) for task in ("Count", "Relation")}
    print("subset", len(rows), counts)
    for row in rows[:2]:
        print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
