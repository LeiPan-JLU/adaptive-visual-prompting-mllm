# -*- coding: utf-8 -*-
"""Aggregate predictions, run paired tests, and draw the paper figures."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = ROOT / "figures"
MODES = ("raw", "grid", "naive_som", "adaptive_som")
LABELS = {"raw": "Raw", "grid": "Grid", "naive_som": "Naive SoM", "adaptive_som": "Adaptive SoM"}
COLORS = {"raw": "#4C72B0", "grid": "#55A868", "naive_som": "#DD8452", "adaptive_som": "#C44E52"}
TASKS = ("Count", "Relation")
TASK_LABEL = {"Count": "2D Counting", "Relation": "2D Spatial Relation"}


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(a, b):
    n = a + b
    if n == 0:
        return 0.0, 1.0
    p = binomtest(min(a, b), n, 0.5, alternative="two-sided").pvalue
    return min(a, b) / n, float(p)


def load():
    meta = {r["id"]: r for r in read_jsonl(ROOT / "data" / "density_meta.jsonl")}
    data = {}
    for mode in MODES:
        path = RESULTS / f"preds_{mode}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        data[mode] = {r["id"]: r for r in read_jsonl(path)}
    ids = sorted(set.intersection(*(set(v) for v in data.values())))
    assert len(ids) > 0, "no overlapping ids"
    return data, meta, ids


def aggregate():
    data, meta, ids = load()
    summary = {"n_total": len(ids), "n_by_task": dict(Counter(meta[i]["task"] for i in ids)), "modes": {}}
    for mode in MODES:
        rows = [data[mode][i] for i in ids]
        entry = {"overall": {}, "latency_mean_sec": float(np.mean([r["latency_sec"] for r in rows])),
                 "parse_fail_rate": float(np.mean([r["pred"] is None for r in rows]))}
        for group in ("Overall", *TASKS):
            subset = rows if group == "Overall" else [r for r in rows if r["task"] == group]
            n = len(subset)
            k = sum(r["correct"] for r in subset)
            lo, hi = wilson(k, n)
            entry[group] = {"n": n, "correct": k, "acc": k / max(1, n), "ci_low": lo, "ci_high": hi,
                            "acc_pct": 100.0 * k / max(1, n)}
        entry["overall"] = entry["Overall"]
        summary["modes"][mode] = entry

    summary["density_buckets"] = {}
    for level in ("Low", "Medium", "High"):
        bucket_ids = [i for i in ids if meta[i]["density_level"] == level]
        summary["density_buckets"][level] = {"n": len(bucket_ids)}
        for mode in MODES:
            k = sum(data[mode][i]["correct"] for i in bucket_ids)
            summary["density_buckets"][level][mode] = k / max(1, len(bucket_ids))
        b = sum(data["adaptive_som"][i]["correct"] and not data["naive_som"][i]["correct"] for i in bucket_ids)
        c = sum((not data["adaptive_som"][i]["correct"]) and data["naive_som"][i]["correct"] for i in bucket_ids)
        _, pval = mcnemar(b, c)
        summary["density_buckets"][level]["p_adaptive_vs_naive"] = pval

    summary["paired_tests"] = {}
    for mode in ("grid", "naive_som", "adaptive_som"):
        b = sum(data[mode][i]["correct"] and not data["raw"][i]["correct"] for i in ids)
        c = sum((not data[mode][i]["correct"]) and data["raw"][i]["correct"] for i in ids)
        q, p = mcnemar(b, c)
        summary["paired_tests"][f"{mode}_vs_raw"] = {"b": b, "c": c, "discordant_ratio": q, "p_value": p}
    b = sum(data["adaptive_som"][i]["correct"] and not data["naive_som"][i]["correct"] for i in ids)
    c = sum((not data["adaptive_som"][i]["correct"]) and data["naive_som"][i]["correct"] for i in ids)
    q, p = mcnemar(b, c)
    summary["paired_tests"]["adaptive_som_vs_naive_som"] = {"b": b, "c": c, "discordant_ratio": q, "p_value": p}
    return summary, data, meta, ids


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def fig1(data, meta, ids):
    high = sorted([i for i in ids if meta[i]["density_level"] == "High"],
                  key=lambda i: (meta[i]["num_boxes"], meta[i]["area_ratio"]), reverse=True)
    picks = high[:2] if len(high) >= 2 else ids[:2]
    fig, axes = plt.subplots(len(picks), 4, figsize=(10.5, 2.5 * len(picks)))
    if len(picks) == 1:
        axes = np.array([axes])
    for r, item_id in enumerate(picks):
        for c, mode in enumerate(MODES):
            ax = axes[r, c]
            ax.imshow(mpimg.imread(ROOT / "prompts_cache" / mode / f"{item_id}.jpg"))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(LABELS[mode], pad=5)
            if c == 0:
                ax.set_ylabel(item_id, fontsize=8)
    fig.tight_layout()
    save(fig, "fig1_method_overview")


def fig2(summary):
    groups = ("Overall", "Count", "Relation")
    x = np.arange(len(groups))
    width = 0.19
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for j, mode in enumerate(MODES):
        vals = [summary["modes"][mode][g]["acc"] for g in groups]
        lows = [summary["modes"][mode][g]["acc"] - summary["modes"][mode][g]["ci_low"] for g in groups]
        highs = [summary["modes"][mode][g]["ci_high"] - summary["modes"][mode][g]["acc"] for g in groups]
        bars = ax.bar(x + (j - 1.5) * width, vals, width, label=LABELS[mode], color=COLORS[mode],
                      yerr=[lows, highs], capsize=2, error_kw={"linewidth": 0.7})
        for bar, value, high in zip(bars, vals, highs):
            ax.text(bar.get_x() + bar.get_width() / 2, value + high + 0.015, f"{value*100:.1f}",
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Overall", TASK_LABEL["Count"], TASK_LABEL["Relation"]])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{v:.0f}" for v in np.arange(0, 101, 20)])
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    save(fig, "fig2_main_accuracy")


def fig3(summary):
    levels = ("Low", "Medium", "High")
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for mode in MODES:
        vals = [summary["density_buckets"][level][mode] * 100 for level in levels]
        ax.plot(levels, vals, marker="o", ms=6, lw=2.0, color=COLORS[mode], label=LABELS[mode])
    med_naive = summary["density_buckets"]["Medium"]["naive_som"] * 100
    med_adapt = summary["density_buckets"]["Medium"]["adaptive_som"] * 100
    ax.annotate(f"{med_naive:.1f}", (1, med_naive), xytext=(8, -14), textcoords="offset points",
                fontsize=7, color=COLORS["naive_som"], fontweight="bold")
    ax.annotate(f"{med_adapt:.1f}", (1, med_adapt), xytext=(8, 8), textcoords="offset points",
                fontsize=7, color=COLORS["adaptive_som"], fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(42, 74)
    ax.set_yticks(np.arange(45, 71, 5))
    ax.set_xlabel("Detected-object density")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    save(fig, "fig3_density_tradeoff")


def _option_text(subset_row, letter):
    if not letter:
        return "—"
    idx = ord(letter) - ord("A")
    choices = subset_row.get("choices", [])
    if 0 <= idx < len(choices):
        return f"{letter} ({choices[idx]})"
    return letter


def fig4(data, meta, ids):
    """Curated 2 Relation + 2 Counting cases, validated against predictions.

    R0036: relation success where Raw/Naive fail and Grid/Adaptive succeed.
    C0061: counting success over Naive (three train proposals, no distractor).
    C0000: counting failure with class-agnostic proposals on walls/furniture.
    R0069: relation failure where 2D boxes cannot express the tabletop layout.
    """
    subset = {r["id"]: r for r in read_jsonl(ROOT / "data" / "eval_subset.jsonl")}
    picks = ["R0036", "C0061", "C0000", "R0069"]
    for item_id in picks:
        if item_id not in subset:
            raise KeyError(item_id)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4))
    axes = axes.ravel()
    for ax, item_id in zip(axes, picks):
        row = data["adaptive_som"][item_id]
        sub = subset[item_id]
        ax.imshow(mpimg.imread(ROOT / "prompts_cache" / "adaptive_som" / f"{item_id}.jpg"))
        ax.set_xticks([])
        ax.set_yticks([])
        if item_id == "R0036":
            label = "Success · Relation (Raw/Naive fail)"
        elif item_id == "C0061":
            label = "Success · Counting (Naive fails)"
        elif item_id == "C0000":
            label = "Failure · Counting (class-agnostic distraction)"
        else:
            label = "Failure · Relation (2D boxes, 3D layout)"
        q = sub["question"]
        gold = _option_text(sub, row["gold"])
        pred = _option_text(sub, row["pred"])
        ax.set_title(f"{item_id} · {label}", fontsize=8)
        ax.set_xlabel(f"Q: {q[:82]}{'...' if len(q) > 82 else ''}\nGold: {gold}  |  Pred: {pred}", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig4_qualitative_cases")


def main():
    setup()
    summary, data, meta, ids = aggregate()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fig1(data, meta, ids)
    fig2(summary)
    fig3(summary)
    fig4(data, meta, ids)
    print(json.dumps({m: summary["modes"][m]["overall"] for m in MODES}, ensure_ascii=False, indent=2))
    print("paired", json.dumps(summary["paired_tests"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
