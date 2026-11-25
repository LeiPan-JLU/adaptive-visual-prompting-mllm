# Method

## Pipeline

1. `data_prep.py` samples 200 Count and 200 Relation questions from the CV-Bench
   2D parquet with seed 42 and saves the question metadata and images.
2. `visual_prompter.py` runs YOLOv8n (confidence 0.25, class-agnostic, at most 20
   boxes) on every image. If no box is returned, a conservative Canny-contour
   fallback supplies proposals for icon-like ADE20K images.
3. For each image, the renderer writes four JPEGs: raw, grid, naive SoM, and
   adaptive SoM.
4. `run_eval.py` runs Qwen2-VL-2B-Instruct on all 1,600 image/question pairs with
   deterministic greedy decoding (`do_sample=False`, `max_new_tokens=32`).
5. `analyze_and_plot.py` computes accuracy, Wilson intervals, density buckets,
   McNemar tests, and the paper figures.

## Density definition

For an image of size `W x H` and proposal boxes `B = {b_i}_{i=1}^N`, let `M` be
the union of all boxes. The union box-area ratio is

```text
rho = |M| / (W * H)
```

The density level is:

```text
Low:    N <= 3 and rho < 0.35
Medium: 4 <= N <= 8 or 0.35 <= rho < 0.60
High:   N >= 9 or rho >= 0.60
```

## Rendering rules

| Level | Box | Label |
|---|---|---|
| Low | translucent box, alpha 0.55, width scaled to short side | small corner tag |
| Medium | 1 px translucent outline, alpha 0.35 | compact in-box tag |
| High | no box; translucent centre dot | micro number tag |

Naive SoM always uses a 3 px red box and an opaque black tag. Grid overlays a
3x3 grid at alpha 0.35 with A1-C3 labels. All annotations are drawn before the
image is passed to the processor; the prompt tells the model that marks are
annotations, not scene content.

## Evaluation

- Metrics: overall, Count, Relation accuracy; 95% Wilson intervals; mean latency;
  parse-failure rate.
- Paired tests: exact McNemar test on the 400 paired predictions, for
  Adaptive vs Raw, Adaptive vs Naive, Grid vs Raw, and Naive vs Raw.
- Density analysis: the same accuracy metrics within Low, Medium, and High
  buckets defined above.
- No test set is used to choose the renderer; the four conditions are fixed
  before inference.
