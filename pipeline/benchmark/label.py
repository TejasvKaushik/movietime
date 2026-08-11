"""Keyboard-driven frame labeller for the filter benchmark.

Uses tkinter rather than cv2.imshow: opencv-python-headless is installed and
wins the import, so cv2 has no GUI here. tkinter is stdlib and adds nothing.

Keys
  1-6   positive, tagged with a clock category (shown on screen)
  y     positive, category unspecified
  n     negative
  s     skip, decide later
  f     re-extract this frame at full resolution and show that instead
  b     back, relabel the previous frame
  q     quit

Why `f` exists: frames are sampled at 640px for the filter, and a wristwatch
that is legible at 1080p can be invisible at 640. Ground truth decided on the
downscaled frame would silently mislabel those as negatives and flatter the
filter's measured recall. Needs the source video, so it is opt-in per frame.

Labels save after every keypress, so an interrupted session resumes where it
stopped.

Usage:
  python pipeline/benchmark/label.py frames/<slug>/filter_scores.json --limit 300
  python pipeline/benchmark/label.py frames/<slug>/manifest.json --order time
"""

import argparse
import json
import random
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sampler"))

CATEGORIES = {
    "1": "analog_wall",
    "2": "digital_display",
    "3": "wristwatch",
    "4": "phone_screen",
    "5": "station_board",
    "6": "prop_stylised",
}

DISPLAY_WIDTH = 1280


def load_frames(source_path, order, limit, seed):
    """Return frame records from either a filter_scores.json or a manifest.json."""
    data = json.loads(Path(source_path).read_text())
    frames = data["frames"]
    scored = "clip_score" in frames[0]

    if order == "score":
        if not scored:
            sys.exit("--order score needs a filter_scores.json, not a manifest")
        # clip_margin over clip_score: raw similarity barely separates on real
        # footage, so ranking by it would mostly shuffle the set randomly.
        frames = sorted(frames, key=lambda f: -max(f["clip_margin"], f["yolo_conf"]))
    elif order == "random":
        frames = random.Random(seed).sample(frames, len(frames))

    return frames[:limit] if limit else frames, data.get("film_slug", "unknown"), scored


class Labeller:
    def __init__(self, frames, slug, scored, out_path, video):
        self.frames = frames
        self.slug = slug
        self.scored = scored
        self.out_path = Path(out_path)
        self.video = video
        self.labels = {}
        self.order = []

        if self.out_path.exists():
            saved = json.loads(self.out_path.read_text())
            self.labels = saved.get("labels", {})
            print(f"resuming: {len(self.labels)} already labelled")

        self.index = 0
        self.photo = None
        self.override = None

        self.root = tk.Tk()
        self.root.title(f"labelling {slug}")
        self.root.configure(bg="black")
        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack()
        self.status = tk.Label(
            self.root, bg="black", fg="#dddddd", font=("Consolas", 10), justify="left"
        )
        self.status.pack(fill="x")

        self.root.bind("<Key>", self.on_key)
        self.advance(0)

    # ---- state -------------------------------------------------------------

    def remaining(self):
        return [i for i, f in enumerate(self.frames) if self.name(f) not in self.labels]

    def name(self, frame):
        return Path(frame["frame_path"]).name

    def save(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        # Spec sketches labels.json as a flat {filename: 0|1} map. Categories are
        # needed for the per-category recall table, so the map is nested under
        # "labels" with the category alongside. Nothing else reads this yet.
        self.out_path.write_text(json.dumps({
            "film_slug": self.slug,
            "labels": self.labels,
        }, indent=2))

    def record(self, value, category=None):
        frame = self.frames[self.index]
        self.labels[self.name(frame)] = {
            "label": value,
            "category": category,
            "frame_path": frame["frame_path"],
            "timecode_seconds": frame["timecode_seconds"],
        }
        self.order.append(self.index)
        self.save()
        self.advance(1)

    def advance(self, step):
        self.override = None
        self.index += step
        while self.index < len(self.frames) and self.name(self.frames[self.index]) in self.labels:
            self.index += 1
        if self.index >= len(self.frames):
            self.finish()
            return
        self.show()

    def finish(self):
        positives = sum(1 for v in self.labels.values() if v["label"] == 1)
        negatives = sum(1 for v in self.labels.values() if v["label"] == 0)
        print(f"\n{len(self.labels)} labelled: {positives} positive, {negatives} negative")
        by_category = {}
        for value in self.labels.values():
            if value["label"] == 1:
                by_category[value["category"] or "untagged"] = (
                    by_category.get(value["category"] or "untagged", 0) + 1
                )
        for category, count in sorted(by_category.items()):
            print(f"  {category:<18} {count}")
        print(f"-> {self.out_path}")
        self.root.destroy()

    # ---- display -----------------------------------------------------------

    def show(self):
        frame = self.frames[self.index]
        path = self.override or Path(frame["frame_path"])
        image = Image.open(path)
        if image.width != DISPLAY_WIDTH:
            ratio = DISPLAY_WIDTH / image.width
            image = image.resize((DISPLAY_WIDTH, int(image.height * ratio)), Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)

        scores = ""
        if self.scored:
            scores = (f"  yolo={frame['yolo_conf']:.3f} "
                      f"clip={frame['clip_score']:.3f} margin={frame['clip_margin']:+.3f}")
        keys = "  ".join(f"{k}={v}" for k, v in CATEGORIES.items())
        self.status.configure(text=(
            f"{len(self.labels)} labelled | {len(self.remaining())} left | "
            f"t={frame['timecode_seconds']:.0f}s{scores}"
            f"{'  [FULL RES]' if self.override else ''}\n"
            f"{keys}\ny=positive  n=negative  s=skip  f=full-res  b=back  q=quit"
        ))

    def full_resolution(self):
        if not self.video:
            print("pass --video to pull full-resolution frames", file=sys.stderr)
            return
        from sample import extract_at

        frame = self.frames[self.index]
        out = Path("frames") / "_fullres" / f"{self.slug}_{self.name(frame)}"
        self.override = extract_at(self.video, frame["timecode_seconds"], out)
        self.show()

    # ---- input -------------------------------------------------------------

    def on_key(self, event):
        key = event.char.lower()
        if key == "q":
            self.finish()
        elif key in CATEGORIES:
            self.record(1, CATEGORIES[key])
        elif key == "y":
            self.record(1)
        elif key == "n":
            self.record(0)
        elif key == "s":
            self.advance(1)
        elif key == "f":
            self.full_resolution()
        elif key == "b" and self.order:
            previous = self.order.pop()
            self.labels.pop(self.name(self.frames[previous]), None)
            self.save()
            self.index = previous
            self.override = None
            self.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", help="filter_scores.json or manifest.json")
    parser.add_argument("--order", choices=("score", "random", "time"), default="score",
                        help="score: highest-ranked first. random: unbiased sample.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="pipeline/benchmark/labels.json")
    parser.add_argument("--video", help="source video, needed for the full-res key")
    args = parser.parse_args()

    frames, slug, scored = load_frames(args.source, args.order, args.limit, args.seed)
    if not frames:
        sys.exit("no frames to label")

    print(f"{len(frames)} frames, order={args.order}")
    Labeller(frames, slug, scored, args.out, args.video).root.mainloop()


if __name__ == "__main__":
    main()
