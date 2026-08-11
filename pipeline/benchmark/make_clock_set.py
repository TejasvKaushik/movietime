"""Generate synthetic clock faces at known times.

This is a best-case ceiling for the VLM stage, not a realistic sample. These
faces are frontal, sharp, high contrast and unoccluded — strictly easier than
any real film frame. A model that cannot read these will not read film frames,
so this is a cheap way to reject a model before spending hours on a real set.

Usage: python pipeline/benchmark/make_clock_set.py [out_dir]
Writes images plus ground_truth.json mapping filename -> {"time": "HH:MM", "type": ...}.
"""

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 512

# Spread across quadrants so a model cannot score well by guessing a common
# hand position. 10:10 is included deliberately: it is the pose used in nearly
# every watch advertisement, so it is the one a model is most likely to answer
# from memory rather than from the image.
ANALOG_TIMES = [
    "01:21", "03:00", "06:30", "09:45",
    "10:10", "12:00", "04:37", "07:52",
    "02:15", "08:05", "11:38", "05:23",
]

DIGITAL_TIMES = ["14:23", "07:05", "23:58", "09:15", "00:42", "18:30"]


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def hand(draw, cx, cy, angle_deg, length, width, fill="black"):
    """Angle measured clockwise from 12 o'clock."""
    rad = math.radians(angle_deg)
    draw.line(
        [(cx, cy), (cx + length * math.sin(rad), cy - length * math.cos(rad))],
        fill=fill,
        width=width,
    )


def analog(time_hhmm):
    hours, minutes = (int(p) for p in time_hhmm.split(":"))
    img = Image.new("RGB", (SIZE, SIZE), "white")
    draw = ImageDraw.Draw(img)
    cx = cy = SIZE // 2
    radius = SIZE // 2 - 24

    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline="black",
        width=6,
    )

    numerals = font(34)
    for n in range(1, 13):
        rad = math.radians(n * 30)
        tx = cx + (radius - 42) * math.sin(rad)
        ty = cy - (radius - 42) * math.cos(rad)
        draw.text((tx, ty), str(n), fill="black", font=numerals, anchor="mm")

        tick = math.radians(n * 30)
        draw.line(
            [
                (cx + (radius - 14) * math.sin(tick), cy - (radius - 14) * math.cos(tick)),
                (cx + radius * math.sin(tick), cy - radius * math.cos(tick)),
            ],
            fill="black",
            width=4,
        )

    # The hour hand advances through the hour, so 06:30 sits halfway between 6 and 7.
    hand(draw, cx, cy, (hours % 12 + minutes / 60) * 30, radius * 0.52, 14)
    hand(draw, cx, cy, minutes * 6, radius * 0.78, 8)
    draw.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill="black")
    return img


def digital(time_hhmm):
    img = Image.new("RGB", (SIZE, SIZE // 2), "#101010")
    draw = ImageDraw.Draw(img)
    draw.text(
        (SIZE // 2, SIZE // 4),
        time_hhmm,
        fill="#ff3b1f",
        font=font(120),
        anchor="mm",
    )
    return img


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "pipeline/benchmark/clock_set")
    out.mkdir(parents=True, exist_ok=True)

    truth = {}
    for kind, times, render in (
        ("analog", ANALOG_TIMES, analog),
        ("digital", DIGITAL_TIMES, digital),
    ):
        for time_hhmm in times:
            name = f"{kind}_{time_hhmm.replace(':', '-')}.png"
            render(time_hhmm).save(out / name)
            truth[name] = {"time": time_hhmm, "type": kind}

    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"{len(truth)} images -> {out}")


if __name__ == "__main__":
    main()
