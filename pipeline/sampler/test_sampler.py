"""Self-check for the sampler. Needs ffmpeg; builds its own test clips.

Run: python pipeline/sampler/test_sampler.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sample import extract_at, hhmmss, require_ffmpeg, sample


def make_clip(path, seconds=10, rate=25, vfr=False):
    filters = ["select='not(mod(n,3))'"] if vfr else []
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x180:rate={rate}"]
        + (["-vf", ",".join(filters), "-fps_mode", "passthrough"] if filters else [])
        + ["-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def main():
    require_ffmpeg()

    assert hhmmss(0) == "00:00:00"
    assert hhmmss(873) == "00:14:33"
    assert hhmmss(3661) == "01:01:01"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        clip = tmp / "Back To The Future.mp4"
        make_clip(clip, seconds=10)
        manifest = json.loads(sample(clip, tmp / "frames", fps=1.0).read_text())

        assert manifest["film_slug"] == "back_to_the_future", manifest["film_slug"]
        frames = manifest["frames"]
        assert 9 <= len(frames) <= 11, f"expected ~10 frames, got {len(frames)}"

        # Timestamps must be ~1s apart and every listed file must exist.
        for i, frame in enumerate(frames):
            assert Path(frame["frame_path"]).exists(), frame["frame_path"]
            assert abs(frame["timecode_seconds"] - i) < 0.5, frame
            assert frame["timecode_hhmmss"] == hhmmss(frame["timecode_seconds"])

        # Rerunning without --force must not re-extract.
        before = {f["frame_path"] for f in frames}
        again = json.loads(sample(clip, tmp / "frames", fps=1.0).read_text())
        assert {f["frame_path"] for f in again["frames"]} == before

        # Never upscale: a 320px source stays 320px even when 640 is requested.
        assert manifest["frame_width"] == 320, manifest["frame_width"]

        # The reason for the select filter: on VFR, reported timestamps must be
        # real source times, not a synthetic 0,1,2,... ladder.
        vfr_clip = tmp / "vfr.mp4"
        make_clip(vfr_clip, seconds=10, vfr=True)
        vfr = json.loads(sample(vfr_clip, tmp / "frames", fps=1.0).read_text())
        stamps = [f["timecode_seconds"] for f in vfr["frames"]]
        assert any(s != int(s) for s in stamps), f"expected real VFR timestamps, got {stamps}"

        # Pass 2 must produce a frame at an exact timestamp from pass 1.
        hires = extract_at(clip, frames[3]["timecode_seconds"], tmp / "hires.jpg")
        assert hires.exists() and hires.stat().st_size > 0

    print("\nall sampler checks passed")


if __name__ == "__main__":
    main()
