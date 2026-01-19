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

    # --- (A) timeline "play" events ---
    # These are card icons on the timeline (both team and opponent sections)
    # Example: <img class="replay_card" data-card="archers" data-t="185" data-s="blue" ...>
    for img in soup.select(".replay_team img.replay_card[data-t][data-card]"):
        t = to_int_or_none(img.get("data-t"))
        card = img.get("data-card")
        side = img.get("data-s")  # "blue" or "red" (or sometimes missing)
        ability = to_int_or_none(img.get("data-ability"))

        if t is None or not card:
            continue
        if card == "_invalid":
            continue

        events.append({
            "type": "play",
            "t": t,
            "side": side,
            "card": card,
            "x": None,
            "y": None,
            "meta": {
                "ability": ability,
                "src": img.get("src"),
            }
        })

    # --- (B) map "place" events ---
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

def write_events_csv(record: Dict[str, Any], out_path: Path) -> None:
    fieldnames = ["replay_id", "battle_time_utc", "t", "type", "side", "card", "x", "y", "ability", "meta_s", "meta_i"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ev in record["events"]:
            meta = ev.get("meta") or {}
            w.writerow({
                "replay_id": record["replay_id"],
                "battle_time_utc": record.get("battle_time_utc"),
                "t": ev.get("t"),
                "type": ev.get("type"),
                "side": ev.get("side"),
                "card": ev.get("card"),
                "x": ev.get("x"),
                "y": ev.get("y"),
                "ability": meta.get("ability"),
                "meta_s": meta.get("s"),
                "meta_i": meta.get("i"),
            })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to replay_*.json file OR directory containing them")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--csv", action="store_true", help="Also write per-replay events CSV")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(input_path)
    if not files:
        raise SystemExit(f"No replay_*.json files found under: {input_path}")

    for fp in files:
        record = parse_replay_json_file(fp)
        rid = record["replay_id"]

        # RL-friendly JSONL (one replay per file; easy to stream)
        jsonl_path = out_dir / f"replay_{rid}.jsonl"
        write_jsonl(record, jsonl_path)

        if args.csv:
            csv_path = out_dir / f"events_{rid}.csv"
            write_events_csv(record, csv_path)

        print(f"[OK] {fp.name} -> {jsonl_path.name} (events={record['event_count']})")

if __name__ == "__main__":
    main()
