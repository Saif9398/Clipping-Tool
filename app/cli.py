"""Headless CLI — same pipeline as the web UI, for testing and scripted runs.

  python -m app.cli <video-url>                 # analyze, list candidates, prompt
  python -m app.cli <video-url> --approve all   # no prompt: render everything
  python -m app.cli <video-url> --approve 1,3,5
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from app import jobs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Clipping Tool: turn long-form videos into vertical clips")
    ap.add_argument("url")
    ap.add_argument("--approve", default=None,
                    help='"all" or comma-separated ranks, e.g. 1,3,5 (skips the prompt)')
    args = ap.parse_args()

    t_start = time.perf_counter()
    job = jobs.create_job(args.url)
    jid = job["id"]
    print(f"job {jid} started")

    job = _wait(jid, ("awaiting_approval",))
    print(f"\n=== {len(job['candidates'])} candidates "
          f"(scored by {job['candidates'][0]['provider'] if job['candidates'] else '-'}) ===")
    for c in job["candidates"]:
        print(f"[{c['rank']:2d}] {c['score']:5.1f}  {_hms(c['start'])}-{_hms(c['end'])} "
              f"({c['duration']:.0f}s)  {c['title']}")
        print(f"      hook: {c['hook']}")
        print(f"      why:  {c['reason']}")

    if args.approve:
        raw = args.approve
    else:
        raw = input("\nRanks to render (e.g. 1,2,3 or 'all'): ").strip()
    if raw.lower() == "all":
        ranks = [c["rank"] for c in job["candidates"]]
    else:
        ranks = [int(x) for x in raw.replace(" ", "").split(",") if x]
    if not ranks:
        print("nothing approved — exiting")
        return 0

    jobs.approve(jid, ranks)
    job = _wait(jid, ("done", "done_with_errors", "error"))

    wall = time.perf_counter() - t_start
    print(f"\nstate={job['state']}  total wall time: {wall/60:.1f} min")
    print("stage timings (s):", job["timings"])
    for clip in job["clips"]:
        print(f"  {clip['file']}  {clip['duration']}s  {clip['frames']} frames")
    if job.get("error"):
        print("errors:", job["error"])
    return 0 if job["state"].startswith("done") else 1


def _wait(jid: str, states: tuple) -> dict:
    last = None
    while True:
        job = jobs.get_job(jid)
        if job["state"] != last:
            last = job["state"]
            print(f"  state -> {last}")
        if last == "error":
            print("ERROR:", job["error"])
            if job.get("trace"):
                print(job["trace"])
            sys.exit(1)
        if last in states:
            return job
        time.sleep(2)


def _hms(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


if __name__ == "__main__":
    sys.exit(main())
