# -*- coding: utf-8 -*-
"""Fill the LaTeX placeholders from results/summary.json (no invented numbers)."""
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
template = ROOT / "paper" / "main.tex"
if not (ROOT / "paper" / "template.tex").exists():
    shutil.copyfile(template, ROOT / "paper" / "template.tex")
text = (ROOT / "paper" / "template.tex").read_text(encoding="utf-8")


def pct(mode, group="Overall"):
    return f"{summary['modes'][mode][group]['acc'] * 100:.1f}"


def lat(mode):
    return f"{summary['modes'][mode]['latency_mean_sec']:.2f}"


def pval(value):
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


peak = 0
log = ROOT / "results" / "run_eval.log"
if log.exists():
    peaks = [int(x) for x in re.findall(r"peak_mem=(\d+)MB", log.read_text(encoding="utf-8", errors="ignore"))]
    peak = max(peaks) if peaks else 0

values = {}
for mode in ("raw", "grid", "naive_som", "adaptive_som"):
    key = {"raw": "RAW", "grid": "GRID", "naive_som": "NAIVE", "adaptive_som": "ADAPT"}[mode]
    values[f"{key}_OVERALL"] = pct(mode)
    values[f"{key}_COUNT"] = pct(mode, "Count")
    values[f"{key}_REL"] = pct(mode, "Relation")
    values[f"{key}_LAT"] = lat(mode)

for level, key in (("Low", "LOW"), ("Medium", "MED"), ("High", "HIGH")):
    bucket = summary["density_buckets"][level]
    values[f"{key}_N"] = str(bucket["n"])
    for mode, mk in (("raw", "RAW"), ("naive_som", "NAIVE"), ("adaptive_som", "ADAPT")):
        values[f"{key}_{mk}"] = f"{bucket[mode] * 100:.1f}"
    values[f"{key}_P"] = pval(bucket["p_adaptive_vs_naive"])

values["P_ADAPT_RAW"] = pval(summary["paired_tests"]["adaptive_som_vs_raw"]["p_value"])
values["P_ADAPT_NAIVE"] = pval(summary["paired_tests"]["adaptive_som_vs_naive_som"]["p_value"])
values["P_NAIVE_RAW"] = pval(summary["paired_tests"]["naive_som_vs_raw"]["p_value"])
values["P_GRID_RAW"] = pval(summary["paired_tests"]["grid_vs_raw"]["p_value"])
values["PEAK_MEM"] = str(peak)

for key, value in values.items():
    text = text.replace(f"@@{key}@@", value)

missing = re.findall(r"@@[A-Z_]+@@", text)
if missing:
    raise SystemExit("unfilled placeholders: " + ", ".join(missing))
(ROOT / "paper" / "main.tex").write_text(text, encoding="utf-8")
print("filled", len(values), "values")
