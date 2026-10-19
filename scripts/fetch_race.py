#!/usr/bin/env python3
"""Fetch FlagOS competition metadata from the public platform API.

The FlagOS web app is a Vue SPA; all race content is served by a JSON API at
`/flagos/api/v1/races[/<rid>]`. This script mirrors that data locally so our
understanding of the rules is reproducible and diffable over time.

Usage
-----
    python scripts/fetch_race.py                 # fetch season 2 + track 2
    python scripts/fetch_race.py --all           # fetch every race on the platform
    python scripts/fetch_race.py --diff          # report changes vs. last snapshot

Output: docs/competition/raw/race_<rid>.json
Exit code 1 if --diff finds a change (useful in CI).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

BASE = "https://flagos.io"
API = f"{BASE}/flagos/api/v1/races"

# The races we actually care about.
SEASON2 = "8q4m2x7p"   # FlagOS 开放计算全球大赛 · 赛季二
TRACK2 = "539vlt2p"    # 赛道二：MiniCPM 模型推理吞吐性能优化
TRACK1 = "782kzq4m"    # 赛道一：SGLang 算子在多款芯片的性能优化

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Lang": "CN",
    "Accept": "application/json",
}

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "competition" / "raw"


def get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        # The platform sits behind a CDN that gzips even without an
        # Accept-Encoding request, so sniff the magic bytes instead of trusting
        # the Content-Encoding header.
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))


def list_races() -> list[dict[str, Any]]:
    return get_json(API)["data"]


def fetch_race(rid: str) -> dict[str, Any]:
    return get_json(f"{API}/{rid}")


def save(rid: str, payload: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"race_{rid}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def strip_volatile(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop fields that change on every request but carry no meaning.

    `metadata.last_updated` is kept: if the organisers edit the page we want to
    know. Only genuinely volatile counters are removed.
    """
    data = json.loads(json.dumps(payload))  # deep copy
    d = data.get("data")
    if isinstance(d, dict):
        info = d.get("race_info")
        if isinstance(info, dict):
            info.pop("team_count", None)  # moves as teams register
        # tabs/children are stable; leave them in
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="fetch every race listed on the platform")
    ap.add_argument("--diff", action="store_true", help="compare with previous snapshot and report changes")
    ap.add_argument("--rid", action="append", default=[], help="fetch a specific race id (repeatable)")
    args = ap.parse_args(argv)

    if args.all:
        rids = [r["rid"] for r in list_races()]
    elif args.rid:
        rids = args.rid
    else:
        rids = [SEASON2, TRACK2, TRACK1]

    changed: list[str] = []
    for rid in rids:
        payload = fetch_race(rid)
        new = strip_volatile(payload)
        path = OUT_DIR / f"race_{rid}.json"

        if args.diff and path.exists():
            old = json.loads(path.read_text(encoding="utf-8-sig"))
            old = strip_volatile(old) if isinstance(old, dict) else old
            if old != new:
                changed.append(rid)
                print(f"CHANGED  {rid}  {path}")
            else:
                print(f"unchanged {rid}")
            continue

        p = save(rid, new)
        title = (new.get("data") or {}).get("title", "?")
        print(f"saved {rid}  {p.stat().st_size:>7} B  {title}")

    if args.diff and changed:
        print(f"\n{len(changed)} race(s) changed: {', '.join(changed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
