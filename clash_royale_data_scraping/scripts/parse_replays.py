#!/usr/bin/env python3
"""
Parse RoyaleAPI replay JSON (from /data/replay) into ordered RL-friendly sequences.

Input: replay_*.json where JSON has at least { "success": true, "html": "<...>" }.
Output:
  - out_dir/replay_<replay_id>.jsonl  (one JSON line per replay with ordered events)
  - out_dir/events_<replay_id>.csv    (flat event table, optional)

Events are merged + sorted by t:
  - play: timeline card play (who played what at time t)
  - place: arena placement marker (x,y coordinate at time t)

Usage:
  python parse_royaleapi_replays.py --input replay_XXXX.json --out out_dir
  python parse_royaleapi_replays.py --input replays_out/ --out parsed_out/ --csv
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup


# -----------------------------
# helpers
# -----------------------------
def to_int_or_none(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    x = str(x).strip()
    if x.lower() == "none" or x == "":
        return None
    try:
        return int(float(x))
    except Exception:
        return None

def text_or_none(el) -> Optional[str]:
    if not el:
        return None
    t = el.get_text(strip=True)
    return t if t else None

def parse_mmss_to_seconds(s: str) -> Optional[int]:
    # "5:11" -> 311
    if not s:
        return None
    m = re.match(r"^\s*(\d+):(\d{2})\s*$", s)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))

def get_replay_id_from_soup(soup: BeautifulSoup) -> Optional[str]:
    # Often stored as: <div class="battle_replay" data-tag="00...">
    root = soup.select_one(".battle_replay[data-tag]")
    if root:
        return root.get("data-tag")
    # fallback: sometimes tag appears in container selectors
    any_tag = soup.select_one('[data-tag]')
    return any_tag.get("data-tag") if any_tag else None

def get_battle_time_utc(soup: BeautifulSoup) -> Optional[str]:
    # Sometimes appears in a tooltip/popover: class battle-timestamp-popup data-content="... UTC"
    # In replay HTML chunk it may or may not exist; we try anyway.
    el = soup.select_one(".battle-timestamp-popup[data-content]")
    return el.get("data-content") if el else None

def get_duration_str_and_seconds(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[int]]:
    end = soup.select_one(".replay_time .marker.end_time")
    dur_str = text_or_none(end)
    dur_sec = parse_mmss_to_seconds(dur_str) if dur_str else None
    return dur_str, dur_sec


# -----------------------------
# core parsing
# -----------------------------
def parse_replay_json_file(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    html = payload.get("html", "")
    soup = BeautifulSoup(html, "html.parser")

    replay_id = get_replay_id_from_soup(soup) or path.stem.replace("replay_", "")
    battle_time_utc = get_battle_time_utc(soup)
    duration_str, duration_seconds = get_duration_str_and_seconds(soup)

    events: List[Dict[str, Any]] = []

    # --- (A) map "place" events ---
    # Example marker:
    # <div class="blue marker ..." data-x="6499" data-y="23499" data-c="archers" data-t="185" data-s="t" ...>
    for mk in soup.select(".replay_map .markers .marker[data-t][data-c]"):
        t = to_int_or_none(mk.get("data-t"))
        card = mk.get("data-c")
        x = to_int_or_none(mk.get("data-x"))
        y = to_int_or_none(mk.get("data-y"))

        # side can be inferred by class (blue/red) or data-s (t/o)
        classes = mk.get("class") or []
        side = "blue" if "blue" in classes else ("red" if "red" in classes else None)

        if t is None or not card:
            continue
        if card == "_invalid":
            continue
        # Some markers have x/y None; those are not placements
        if x is None or y is None:
            continue

        events.append({
            "type": "place",
            "t": t,
            "side": side,
            "card": card,
            "x": x,
            "y": y,
            "meta": {
                "i": to_int_or_none(mk.get("data-i")),
                "s": mk.get("data-s"),  # often "t" (team) / "o" (opponent)
            }
        })

    # ---- sort in-time, stable by type (plays first) ----
    # If same t: prefer "play" before "place" so the agent sees intent before location,
    # but you can flip this if you want.
    type_rank = {"play": 0, "place": 1}
    events.sort(key=lambda e: (e["t"], type_rank.get(e["type"], 9), e.get("card") or ""))

    # ---- build final RL-friendly record ----
    record = {
        "replay_id": replay_id,
        "battle_time_utc": battle_time_utc,
        "duration_str": duration_str,
        "duration_seconds": duration_seconds,
        "event_count": len(events),
        "events": events,
        # keep raw for debugging if you want (comment out if too big)
        # "raw": {"success": payload.get("success"), "status": payload.get("status")},
    }
    return record

def parse_directory(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_files = sorted(input_dir.rglob("*.json"))

    if not replay_files:
        print("No replay JSON files found.")
        return

    print(f"Found {len(replay_files)} replay files")

    for replay_path in replay_files:
        if replay_path.name == "manifest.json":
            continue

        parsed = parse_replay_json_file(replay_path)

        out_path = output_dir / replay_path.with_suffix(".jsonl").name
        write_jsonl(parsed, out_path)

        print(f"Parsed → {out_path.name}")


# -----------------------------
# IO / CLI
# -----------------------------
def iter_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.glob("replay_*.json"))
    return files

def write_jsonl(record: Dict[str, Any], out_path: Path) -> None:
    # One JSON object per line
    out_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing raw replay JSON files")
    ap.add_argument("--output_dir", required=True, help="Directory to save parsed RL-ready files")
    args = ap.parse_args()

    parse_directory(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
