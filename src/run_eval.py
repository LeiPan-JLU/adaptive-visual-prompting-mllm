# -*- coding: utf-8 -*-
"""Run Qwen2-VL-2B-Instruct on the four visual-prompt variants."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "data" / "eval_subset.jsonl"
CACHE = ROOT / "prompts_cache"
RESULTS = ROOT / "results"
MODEL_DIR = ROOT / "models" / "Qwen2-VL-2B-Instruct"
MODES = ("raw", "grid", "naive_som", "adaptive_som")
LETTERS = "ABCDEF"


def read_rows(limit: int | None = None):
    rows = [json.loads(line) for line in SUBSET.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.add(json.loads(line)["id"])
            except Exception:
                continue
    return out


def parse_answer(text: str) -> str | None:
    text = text.strip()
    patterns = [
        r"Answer\s*[:：]\s*\(?([A-F])\)?",
        r"\(([A-F])\)",
        r"\b([A-F])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def build_prompt(row: dict, mode: str) -> str:
    mode_hint = {
        "raw": "",
        "grid": "A semi-transparent 3x3 grid with cell labels A1-C3 is overlaid for spatial reference. ",
        "naive_som": "Numbered boxes mark detected objects; the numbers are annotations, not part of the scene. ",
        "adaptive_som": "Small numbered marks mark detected objects; the marks are annotations, not part of the scene. ",
    }[mode]
    return (
        "You are a careful visual question answering assistant. "
        f"{mode_hint}"
        "Answer the multiple-choice question using the image. "
        "Return the option letter exactly in the final line, formatted as 'Answer: X', where X is one of A, B, C, D, E, F.\n\n"
        f"{row['prompt']}"
    )


def load_model():
    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR), local_files_only=True,
        torch_dtype=torch.float16, attn_implementation="sdpa",
    )
    model.to("cuda:0")
    model.eval()
    return processor, model


def run_one(processor, model, row, mode):
    image_path = CACHE / mode / f"{row['id']}.jpg"
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": build_prompt(row, mode)},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda:0")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    new_tokens = generated[:, inputs.input_ids.shape[1]:]
    output = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output.strip(), latency, int(inputs.input_ids.shape[1]), int(new_tokens.shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["all", *MODES])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.limit)
    modes = MODES if args.mode == "all" else (args.mode,)
    processor, model = load_model()
    print("model loaded", flush=True)
    for mode in modes:
        out_path = RESULTS / f"preds_{mode}.jsonl"
        done = existing(out_path)
        todo = [r for r in rows if r["id"] not in done]
        print(f"{mode}: {len(todo)} pending / {len(rows)}", flush=True)
        with out_path.open("a", encoding="utf-8") as f:
            for i, row in enumerate(todo, 1):
                try:
                    output, latency, n_in, n_out = run_one(processor, model, row, mode)
                    pred = parse_answer(output)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    output, pred, latency, n_in, n_out = "OOM", None, 0.0, 0, 0
                except Exception as exc:  # noqa: BLE001
                    output, pred, latency, n_in, n_out = f"ERROR: {exc}", None, 0.0, 0, 0
                record = {
                    "id": row["id"],
                    "task": row["task"],
                    "mode": mode,
                    "pred": pred,
                    "gold": row["answer"],
                    "correct": bool(pred == row["answer"]),
                    "raw_output": output,
                    "latency_sec": round(latency, 4),
                    "input_tokens": n_in,
                    "output_tokens": n_out,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"{mode} [{i}/{len(todo)}] {row['id']} pred={pred} gold={row['answer']} "
                          f"{latency:.2f}s", flush=True)
                if i % 50 == 0:
                    torch.cuda.empty_cache()
        records = [json.loads(x) for x in out_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        acc = sum(r["correct"] for r in records) / max(1, len(records))
        parse_fail = sum(r["pred"] is None for r in records) / max(1, len(records))
        print(f"{mode}: acc={acc:.4f} n={len(records)} parse_fail={parse_fail:.4f} "
              f"peak_mem={torch.cuda.max_memory_allocated()/1024**2:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
