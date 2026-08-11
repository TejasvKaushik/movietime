"""Cheap filter: score every frame for clock-likeness with YOLO and CLIP.

The spec describes a cascade — run YOLO, send only its rejects to CLIP — which
saves compute in production. This scores every frame with both models instead,
because a frame CLIP never saw has no CLIP score, and a model's PR curve cannot
be computed over frames it never scored. Recording raw scores for everything
once turns threshold calibration into a re-read of this file rather than a
re-run of inference. Add the cascade when CLIP time actually hurts.

`passed` is therefore derived, not measured: it is the thresholds applied to
stored scores. Re-thresholding needs no GPU and no second pass.

Two CLIP scores are recorded per frame:
  clip_score  — max cosine similarity over the clock prompts (the spec's metric)
  clip_margin — that, minus the best non-clock prompt

Raw similarity is poorly calibrated: some frames score high against every
prompt. The margin controls for that. Which one discriminates better is a
question for the benchmark, so both are stored.

Usage:
  python pipeline/filter/filter_frames.py frames/<slug>/manifest.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

CLOCK_PROMPTS = [
    "an analog clock face showing a time",
    "a digital clock or timer display",
    "a wristwatch showing the time",
    "a wall clock",
    "a phone or screen displaying the time",
    "a train station departure board",
    "a countdown timer",
    "clock hands pointing to a time",
    "a microwave or oven display showing numbers",
    "a VCR or cassette player display",
]

# Deliberately common film content, so a frame that merely "looks like a photo"
# does not inflate the clock score.
OTHER_PROMPTS = [
    "a person's face in close up",
    "two people talking in a room",
    "an empty room interior",
    "a city street at night",
    "a car on a road",
    "a landscape or sky",
    "hands holding an object",
    "a dark or black frame",
]

COCO_CLOCK_CLASS = 74


def load_models(clip_name):
    """Import heavy deps here so --help and argument errors stay instant."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from ultralytics import YOLO

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO("yolov8n.pt")
    clip = CLIPModel.from_pretrained(clip_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_name)

    # Text embeddings never change per frame, so they are computed once.
    prompts = CLOCK_PROMPTS + OTHER_PROMPTS
    tokens = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    text_features = clip.get_text_features(**tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return torch, device, yolo, clip, processor, text_features


def score_batch(torch, device, yolo, clip, processor, text_features, paths):
    """Return (yolo_confs, clip_scores, clip_margins, yolo_seconds, clip_seconds)."""
    from PIL import Image

    # conf=0.01, not the ultralytics default of 0.25: low-confidence detections
    # are exactly what a recall-first filter must be able to threshold on later.
    started = time.perf_counter()
    detections = yolo([str(p) for p in paths], conf=0.01, verbose=False)
    yolo_confs = []
    for result in detections:
        boxes = result.boxes
        clock = boxes.conf[boxes.cls == COCO_CLOCK_CLASS]
        yolo_confs.append(float(clock.max()) if len(clock) else 0.0)
    yolo_seconds = time.perf_counter() - started

    started = time.perf_counter()
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    features = clip.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    similarity = features @ text_features.T

    clock_best = similarity[:, : len(CLOCK_PROMPTS)].max(dim=1).values
    other_best = similarity[:, len(CLOCK_PROMPTS) :].max(dim=1).values
    clip_seconds = time.perf_counter() - started

    return (
        yolo_confs,
        clock_best.tolist(),
        (clock_best - other_best).tolist(),
        yolo_seconds,
        clip_seconds,
    )


def run(manifest_path, batch_size, yolo_threshold, clip_threshold, clip_name, limit=None):
    manifest = json.loads(Path(manifest_path).read_text())
    frames = manifest["frames"][:limit] if limit else manifest["frames"]
    paths = [Path(f["frame_path"]) for f in frames]

    torch, device, yolo, clip, processor, text_features = load_models(clip_name)
    print(f"device={device}  frames={len(paths)}  batch={batch_size}", flush=True)

    records = []
    totals = {"yolo": 0.0, "clip": 0.0}
    started = time.perf_counter()

    for offset in range(0, len(paths), batch_size):
        chunk = paths[offset : offset + batch_size]
        confs, scores, margins, yolo_seconds, clip_seconds = score_batch(
            torch, device, yolo, clip, processor, text_features, chunk
        )
        totals["yolo"] += yolo_seconds
        totals["clip"] += clip_seconds

        for frame, conf, score, margin in zip(frames[offset:], confs, scores, margins):
            by_yolo = conf >= yolo_threshold
            by_clip = score >= clip_threshold
            records.append({
                "frame_path": frame["frame_path"],
                "timecode_seconds": frame["timecode_seconds"],
                "passed": by_yolo or by_clip,
                "stage": "yolo" if by_yolo else ("clip" if by_clip else "none"),
                "yolo_conf": round(conf, 4),
                "clip_score": round(score, 4),
                "clip_margin": round(margin, 4),
            })

        done = offset + len(chunk)
        if done % (batch_size * 10) == 0 or done == len(paths):
            rate = done / (time.perf_counter() - started)
            print(f"  {done}/{len(paths)}  {rate:.1f} frames/s", flush=True)

    elapsed = time.perf_counter() - started
    passed = sum(r["passed"] for r in records)

    out = Path(manifest_path).parent / "filter_scores.json"
    out.write_text(json.dumps({
        "film_slug": manifest["film_slug"],
        "device": device,
        "clip_model": clip_name,
        "yolo_model": "yolov8n.pt",
        "clock_prompts": CLOCK_PROMPTS,
        "other_prompts": OTHER_PROMPTS,
        "thresholds": {"yolo_conf": yolo_threshold, "clip_score": clip_threshold},
        "batch_size": batch_size,
        "timing": {
            "total_seconds": round(elapsed, 2),
            "yolo_seconds": round(totals["yolo"], 2),
            "clip_seconds": round(totals["clip"], 2),
            "frames_per_second": round(len(records) / elapsed, 2),
        },
        "frames": records,
    }, indent=2))

    print(f"\n{passed}/{len(records)} passed ({passed / len(records):.1%}) "
          f"at yolo>={yolo_threshold} clip>={clip_threshold}")
    print(f"yolo {totals['yolo']:.0f}s  clip {totals['clip']:.0f}s  "
          f"total {elapsed:.0f}s  {len(records) / elapsed:.1f} frames/s")
    print(f"-> {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("manifest")
    parser.add_argument("--batch", type=int, default=int(os.environ.get("BATCH_SIZE_FILTER", 32)))
    parser.add_argument("--yolo-conf", type=float,
                        default=float(os.environ.get("FILTER_YOLO_CONF", 0.15)))
    parser.add_argument("--clip-threshold", type=float,
                        default=float(os.environ.get("FILTER_CLIP_THRESHOLD", 0.23)))
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--limit", type=int, help="score only the first N frames")
    args = parser.parse_args()

    if not Path(args.manifest).exists():
        sys.exit(f"no manifest at {args.manifest}")

    run(args.manifest, args.batch, args.yolo_conf, args.clip_threshold,
        args.clip_model, args.limit)


if __name__ == "__main__":
    main()
