"""Measure VLM time-reading accuracy against the synthetic clock set.

The spec benchmarks the filter's recall but never the VLM's read accuracy, even
though the VLM's output is what lands in the database. This closes that gap.

Scoring differs by clock type on purpose. A digital display states the hour
unambiguously, so it must match exactly. An analog face cannot distinguish
01:21 from 13:21 — no visual signal exists — so an hour off by exactly 12 is
counted correct. Penalising that would measure a limit of the medium, not the
model.

Model names carry their backend as a prefix: "ollama:qwen2.5vl:3b" or
"gemini:gemini-2.5-flash". Gemini needs GEMINI_API_KEY in the environment.

Usage: python pipeline/benchmark/vlm_spotcheck.py [model] [image_dir]
"""

import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OLLAMA = "http://localhost:11434/api/generate"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = (
    "Look at this image. If there is a clock, watch, digital display, or any "
    "visible time indicator in the image, state the time it shows in HH:MM "
    "format (24-hour). If there is no readable time in the image, respond with "
    "NONE. Respond with only the time in HH:MM format or the word NONE. "
    "Nothing else."
)

TIME_RE = re.compile(r"\b(\d{1,2})[:.](\d{2})\s*(am|pm)?", re.I)
BARE_MERIDIEM_RE = re.compile(r"\b(\d{1,2})\s*(am|pm)\b", re.I)


def parse_time(raw):
    """Pull an HH:MM 24-hour string out of a model response, or None.

    Shared with the real VLM stage later, so it handles what models actually
    emit: '3:05', '3pm', '01.21', and prose wrapped around any of those.
    """
    if raw is None:
        return None

    match = TIME_RE.search(raw)
    if match:
        hours, minutes, meridiem = int(match[1]), int(match[2]), match[3]
    else:
        match = BARE_MERIDIEM_RE.search(raw)
        if not match:
            return None
        hours, minutes, meridiem = int(match[1]), 0, match[2]

    if meridiem:
        meridiem = meridiem.lower()
        if hours == 12:
            hours = 0
        if meridiem == "pm":
            hours += 12

    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        return None
    return f"{hours:02d}:{minutes:02d}"


def post(url, payload, timeout=600, attempts=5):
    """POST JSON, retrying on rate limits and transient upstream failures.

    Free-tier quotas are low enough that a plain loop over a few dozen images
    will trip 429 partway through, so backoff is required rather than polish.
    """
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    for attempt in range(attempts):
        try:
            # Timed inside the loop so backoff sleeps stay out of the latency
            # figure — otherwise a throttled run reports invented slowness.
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response), time.perf_counter() - started
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or attempt == attempts - 1:
                raise
            # Honour Retry-After when the server sends it; otherwise back off.
            wait = float(e.headers.get("Retry-After") or 0) or min(2 ** attempt * 5, 60)
            print(f"    {e.code}, retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)


def ask_ollama(model, image_b64):
    body, seconds = post(OLLAMA, {
        "model": model,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0},
    })
    return body.get("response", "").strip(), seconds


def ask_gemini(model, image_b64):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set. Get one at https://aistudio.google.com/apikey")

    body, seconds = post(f"{GEMINI.format(model=model)}?key={key}", {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "image/png", "data": image_b64}},
        ]}],
        "generationConfig": {"temperature": 0},
    })
    try:
        return body["candidates"][0]["content"]["parts"][0]["text"].strip(), seconds
    except (KeyError, IndexError):
        # A safety block or an empty candidate is a legitimate "unreadable",
        # not a crash — the pipeline must survive it.
        return "", seconds


BACKENDS = {"ollama": ask_ollama, "gemini": ask_gemini}


def ask(model, image_path):
    backend, _, name = model.partition(":")
    if backend not in BACKENDS:
        sys.exit(f"unknown backend {backend!r}; expected one of {sorted(BACKENDS)}")

    return BACKENDS[backend](name, base64.b64encode(image_path.read_bytes()).decode())


def minutes_of(time_hhmm):
    return int(time_hhmm[:2]) * 60 + int(time_hhmm[3:])


def is_correct(predicted, actual, kind):
    if predicted is None:
        return False
    if predicted == actual:
        return True
    # Analog faces carry no AM/PM information; a 12-hour offset is not an error.
    return kind == "analog" and abs(minutes_of(predicted) - minutes_of(actual)) == 720


def main():
    # Pinned, not an alias like gemini-flash-latest: results are timestamped and
    # compared across runs, so the model must not shift underneath them.
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini:gemini-3.6-flash"
    image_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "pipeline/benchmark/clock_set")

    truth = json.loads((image_dir / "ground_truth.json").read_text())
    results = []

    for name, meta in sorted(truth.items()):
        # One unreachable frame must not discard the whole run's results.
        error = None
        try:
            raw, seconds = ask(model, image_dir / name)
        except urllib.error.HTTPError as e:
            raw, seconds, error = None, 0.0, f"{e.code} {e.read().decode()[:200]}"
        except urllib.error.URLError as e:
            raw, seconds, error = None, 0.0, str(e.reason)

        predicted = parse_time(raw)
        correct = is_correct(predicted, meta["time"], meta["type"])
        # Off-by-a-few-minutes means the model is reading the dial imprecisely.
        # Wildly wrong means it is not reading it at all. Very different problems.
        drift = None
        if predicted:
            gap = abs(minutes_of(predicted) - minutes_of(meta["time"])) % 720
            drift = min(gap, 720 - gap)

        results.append({
            "image": name,
            "type": meta["type"],
            "expected": meta["time"],
            "predicted": predicted,
            "raw": raw,
            "correct": correct,
            "drift_minutes": drift,
            "seconds": round(seconds, 2),
            "error": error,
        })
        status = "ERR " if error else ("OK  " if correct else "MISS")
        print(
            f"{status}  {name:<24} "
            f"expected {meta['time']}  got {predicted or '-':<5} "
            f"({drift if drift is not None else '-'} min off)  {seconds:.1f}s"
            + (f"  [{error[:60]}]" if error else "")
        )

    print()
    summary = {}
    for kind in ("analog", "digital"):
        subset = [r for r in results if r["type"] == kind]
        if not subset:
            continue
        # A transport failure is not the model being wrong, so accuracy is
        # measured over frames that actually got an answer, and errors are
        # reported alongside rather than folded in.
        answered = [r for r in subset if not r["error"]]
        errors = len(subset) - len(answered)
        if not answered:
            summary[kind] = {"n": len(subset), "answered": 0, "errors": errors}
            print(f"{kind:<8} no successful reads ({errors} errors)")
            continue

        hits = sum(r["correct"] for r in answered)
        within5 = sum(
            1 for r in answered if r["drift_minutes"] is not None and r["drift_minutes"] <= 5
        )
        summary[kind] = {
            "n": len(subset),
            "answered": len(answered),
            "errors": errors,
            "correct": hits,
            "accuracy": round(hits / len(answered), 3),
            "within_5_min": within5,
            "mean_seconds": round(sum(r["seconds"] for r in answered) / len(answered), 2),
        }
        print(
            f"{kind:<8} {hits}/{len(answered)} exact  "
            f"{within5}/{len(answered)} within 5 min  "
            f"{summary[kind]['mean_seconds']}s/frame"
            + (f"  ({errors} errors)" if errors else "")
        )

    out_dir = Path("pipeline/benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = out_dir / f"vlm_spotcheck_{model.replace(':', '-')}_{stamp}.json"
    report.write_text(json.dumps(
        {"model": model, "prompt": PROMPT, "summary": summary, "results": results},
        indent=2,
    ))
    print(f"\n-> {report}")


if __name__ == "__main__":
    main()
