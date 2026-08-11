"""Extract frames from a video at a fixed rate, preserving source timecodes.

Two entry points, matching the two-pass design:

  sample()     — pass 1, every Nth second at filter resolution (small, ~250MB/film)
  extract_at() — pass 2, one frame at full resolution, for frames the filter kept

Frame selection uses the `select` filter with `-fps_mode passthrough` rather
than the more obvious `fps` filter. `fps` resamples onto a synthetic constant
timeline and rewrites PTS, so on variable-frame-rate sources it reports
timestamps for frames that do not exist. Measured on a VFR test clip, `fps=1`
claimed frames at 0,1,2,3 while the real ones sat at 0,1.08,2.16,3.24. Pass 2
seeks back to these timestamps, so a fabricated one would re-extract a
different frame than the filter approved.

Usage:
  python pipeline/sampler/sample.py VIDEO [--fps 1.0] [--width 640] [--out frames]
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def require_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH. Install with: scoop install ffmpeg")


def probe(video):
    """Source width, height and duration in seconds."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration",
            "-of", "json", str(video),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    return stream["width"], stream["height"], float(data["format"]["duration"])


def hhmmss(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def sample(video, out_root, fps=1.0, width=640, quality=5, force=False):
    """Extract frames at `fps` into out_root/<slug>/, returning the manifest path."""
    video = Path(video)
    slug = re.sub(r"[^a-z0-9]+", "_", video.stem.lower()).strip("_")
    out_dir = Path(out_root) / slug
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists() and not force:
        print(f"{slug}: manifest exists, skipping (use --force to redo)")
        return manifest_path

    source_width, _, duration = probe(video)
    # Never upscale — a 480p source gains nothing from a 640px target.
    target_width = min(width, source_width)
    out_dir.mkdir(parents=True, exist_ok=True)

    interval = 1.0 / fps
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(video),
        "-vf", (
            f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',"
            f"scale={target_width}:-2,showinfo"
        ),
        "-fps_mode", "passthrough",
        "-q:v", str(quality),
        str(out_dir / "%08d.jpg"),
    ]

    print(f"{slug}: {duration:.0f}s source, sampling {fps}fps at {target_width}px wide")

    # Stream stderr rather than buffering it: showinfo emits one line per frame,
    # and a 3h film at 1fps produces ~10,800 of them.
    timecodes = []
    process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    for line in process.stderr:
        match = PTS_RE.search(line)
        if match:
            timecodes.append(float(match[1]))
            if len(timecodes) % 500 == 0:
                print(f"  {len(timecodes)} frames  ({timecodes[-1]:.0f}s)")
    if process.wait() != 0:
        sys.exit(f"ffmpeg failed on {video}")

    # ffmpeg numbers output files from 1. A frame ffmpeg reported but failed to
    # write (corrupt source) is dropped rather than allowed to crash the run.
    frames, skipped = [], 0
    for index, seconds in enumerate(timecodes, start=1):
        frame_path = out_dir / f"{index:08d}.jpg"
        if not frame_path.exists() or frame_path.stat().st_size == 0:
            skipped += 1
            continue
        frames.append({
            "frame_path": str(frame_path).replace("\\", "/"),
            "timecode_seconds": round(seconds, 3),
            "timecode_hhmmss": hhmmss(seconds),
        })

    manifest_path.write_text(json.dumps({
        "film_slug": slug,
        "source": str(video).replace("\\", "/"),
        "duration_seconds": round(duration, 3),
        "sample_fps": fps,
        "frame_width": target_width,
        "frames": frames,
    }, indent=2))

    print(f"{slug}: {len(frames)} frames -> {manifest_path}"
          + (f"  ({skipped} unwritable, skipped)" if skipped else ""))
    return manifest_path


def extract_at(video, seconds, out_path, quality=2):
    """Pass 2: pull a single full-resolution frame at an exact source timestamp.

    -ss before -i seeks by keyframe first and then decodes forward, so this is
    fast and still lands on the requested frame.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{seconds:.3f}",
            "-i", str(video),
            "-frames:v", "1",
            "-q:v", str(quality),
            str(out_path),
        ],
        check=True,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=640,
                        help="filter-stage width; CLIP resizes to 224 regardless")
    parser.add_argument("--out", default="frames")
    parser.add_argument("--quality", type=int, default=5, help="ffmpeg -q:v, 2=best 31=worst")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    require_ffmpeg()
    sample(args.video, args.out, args.fps, args.width, args.quality, args.force)


if __name__ == "__main__":
    main()
