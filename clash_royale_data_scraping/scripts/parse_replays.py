#!/usr/bin/env python3
"""
Parse RoyaleAPI replay JSON (from /data/replay) into ordered BC-friendly sequences.

Supports two input formats:
  (A) Old:  { "success": true, "html": "<...>" }
  (B) New:  { "meta": {...}, "data": { "success": true, "html": "<...>" } }

Outputs:
  - output_dir/replay_<replay_id>.jsonl   (1 line = 1 replay summary record w/ parsed events)
  - output_dir/bc_<replay_id>.jsonl      (many lines, 1 line = 1 BC training sample)

BC sample format (per decision tick):
{
  "replay_id": "...",
  "t": 1500,                 # ms from start (tick start)
  "tick_ms": 500,
  "player": "team",          # "team" is the scraped player in meta
  "deck": [...8 cards...],   # normalized
  "opp_deck": [...8 cards...],
  "history": [
      {"p": "me", "card": "knight", "t": 1200},
      {"p": "opp","card": "fireball","t": 1300},
      ...
  ],
  "label": "NOOP" or "<card_id>"
}

Notes:
- "NOOP" is included by creating decision ticks at fixed intervals.
- Evolution suffixes like "-ev1" are stripped everywhere so deck/actions match.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup


# -----------------------------
# arena coordinate mapping
# -----------------------------

ARENA_COLS = 20
ARENA_ROWS = 34

def arena_xy_to_grid(x: int, y: int) -> tuple[int, int] | tuple[None, None]:
    """
    Convert RoyaleAPI arena coords (data-x, data-y) into (row, col) on a 34x20 grid.

    Observed from RoyaleAPI HTML:
      - left is multiples of 5%   => 20 columns
      - top  is multiples of 100/34% => 34 rows
      - data-x and data-y are in ~1000-unit steps with a +500 offset
      - x increases to the LEFT, y increases UP

    Returns:
      (row, col) as 0-based ints:
        row in [0, 33], col in [0, 19]
      or (None, None) if invalid.
    """
    if x is None or y is None:
        return None, None

    try:
        # Derived directly from the HTML percentage steps you pasted.
        col = int(round(18.5 - (x / 1000.0)))
        row = int(round(32.5 - (y / 1000.0)))
    except Exception:
        return None, None

    # Clamp to grid bounds (just in case)
    if col < 0: col = 0
    if col > (ARENA_COLS - 1): col = ARENA_COLS - 1
    if row < 0: row = 0
    if row > (ARENA_ROWS - 1): row = ARENA_ROWS - 1

    return row, col


def arena_xy_to_placement_index(x: int, y: int) -> int | None:
    """
    Flatten (row, col) into a single 0-based placement index:
        idx = row * 20 + col

    This gives you a single categorical location token in [0, 679].
    """
    row, col = arena_xy_to_grid(x, y)
    if row is None or col is None:
        return None
    return row * ARENA_COLS + col


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


def normalize_card_id(card: Optional[str]) -> Optional[str]:
    if not card:
        return card
    # strip evolution suffixes like -ev1, -ev2, etc.
    return re.sub(r"-ev\d+$", "", card)


def get_replay_id_from_soup(soup: BeautifulSoup) -> Optional[str]:
    root = soup.select_one(".battle_replay[data-tag]")
    if root:
        return root.get("data-tag")
    any_tag = soup.select_one('[data-tag]')
    return any_tag.get("data-tag") if any_tag else None


def get_battle_time_utc(soup: BeautifulSoup) -> Optional[str]:
    el = soup.select_one(".battle-timestamp-popup[data-content]")
    return el.get("data-content") if el else None


def get_duration_str_and_seconds(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[int]]:
    end = soup.select_one(".replay_time .marker.end_time")
    dur_str = text_or_none(end)
    dur_sec = parse_mmss_to_seconds(dur_str) if dur_str else None
    return dur_str, dur_sec


def load_payload(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (meta, data_payload) where:
      meta may be {}
      data_payload contains at least html if present
    """
    payload = json.loads(path.read_text(encoding="utf-8"))

    # New format: {"meta": {...}, "data": {...}}
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], dict):
        meta = payload.get("meta") or {}
        data = payload["data"]
        return meta, data

    # Old format: {"html": "..."}
    return {}, payload


def infer_team_side_from_place_events(place_events: List[Dict[str, Any]]) -> Optional[str]:
    """
    Try to infer which color ("blue"/"red") corresponds to team ("t") using map markers.
    Your existing parser stores mk.get("data-s") into meta["s"] (often "t" or "o").
    """
    counts = {"blue": 0, "red": 0}
    for e in place_events:
        if e.get("type") != "place":
            continue
        meta_s = (e.get("meta") or {}).get("s")
        side = e.get("side")
        if meta_s == "t" and side in ("blue", "red"):
            counts[side] += 1

    if counts["blue"] == 0 and counts["red"] == 0:
        return None
    return "blue" if counts["blue"] >= counts["red"] else "red"


def pick_first_non_null_deck(decks: Any) -> List[str]:
    """
    meta.team_decks / meta.opponent_decks is typically a list aligned with tags.
    We pick the first non-null deck.
    """
    if not decks:
        return []
    if isinstance(decks, list):
        for d in decks:
            if isinstance(d, list) and len(d) >= 8:
                return [normalize_card_id(x) for x in d[:8] if x]
    return []


# -----------------------------
# parsing replays
# -----------------------------
def parse_replay_json_file(path: Path) -> Dict[str, Any]:
    meta, data = load_payload(path)
    html = data.get("html", "") if isinstance(data, dict) else ""
    soup = BeautifulSoup(html, "html.parser")

    replay_id = get_replay_id_from_soup(soup) or path.stem.replace("replay_", "")
    battle_time_utc = get_battle_time_utc(soup)
    duration_str, duration_seconds = get_duration_str_and_seconds(soup)

    events: List[Dict[str, Any]] = []

    # --- (A) PLAY events: timeline cards ---
    # Example:
    # <img class="replay_card" data-card="knight" data-t="3226" data-s="red" data-ability="0" ...>
    for img in soup.select("img.replay_card[data-card][data-t][data-s]"):
        t = to_int_or_none(img.get("data-t"))
        card_raw = img.get("data-card")
        ability = img.get("data-ability")
        side = img.get("data-s")  # "blue" or "red"

        card = normalize_card_id(card_raw)

        if t is None or not card:
            continue
        if card == "_invalid":
            continue
        # ability=1 is champion ability / not a normal card play; skip for now
        if ability is not None and str(ability).strip() == "1":
            continue
        if side not in ("blue", "red"):
            continue

        events.append({
                            "type": "play",
                            "t": t,
                            "side": side,
                            "card": card,
                            "meta": {
                                "ability": to_int_or_none(ability),
                            }
                        })

    # --- (B) PLACE events: map markers (your existing logic) ---
    # Example marker:
    # <div class="blue marker ..." data-x="6499" data-y="23499" data-c="archers" data-t="185" data-s="t" ...>
    for mk in soup.select(".replay_map .markers .marker[data-t][data-c]"):
        t = to_int_or_none(mk.get("data-t"))
        card = normalize_card_id(mk.get("data-c"))
        x = to_int_or_none(mk.get("data-x"))
        y = to_int_or_none(mk.get("data-y"))

        classes = mk.get("class") or []
        side = "blue" if "blue" in classes else ("red" if "red" in classes else None)

        if t is None or not card:
            continue
        if card == "_invalid":
            continue
        if x is None or y is None:
            continue

        row, col = arena_xy_to_grid(x, y)
        placement_index = arena_xy_to_placement_index(x, y)

        events.append({
                            "type": "place",
                            "t": t,
                            "side": side,
                            "card": card,
                            "x": x,
                            "y": y,
                            "row": row,
                            "col": col,
                            "placement_index": placement_index,
                            "meta": {
                                "i": to_int_or_none(mk.get("data-i")),
                                "s": mk.get("data-s"),
                            }
                        })

    # ---- sort by time, with plays first if same t ----
    type_rank = {"play": 0, "place": 1}
    events.sort(key=lambda e: (e["t"], type_rank.get(e["type"], 9), e.get("card") or ""))

    # ---- attach decks from meta (your downloader stores these) ----
    team_deck = pick_first_non_null_deck((meta or {}).get("team_decks"))
    opp_deck = pick_first_non_null_deck((meta or {}).get("opponent_decks"))

    record = {
        "replay_id": replay_id,
        "battle_time_utc": battle_time_utc,
        "duration_str": duration_str,
        "duration_seconds": duration_seconds,
        "event_count": len(events),
        "events": events,
        "meta": {
            "player_tag": (meta or {}).get("player_tag"),
            "team_tags": (meta or {}).get("team_tags"),
            "opponent_tags": (meta or {}).get("opponent_tags"),
            "team_deck": team_deck,
            "opponent_deck": opp_deck,
            "source_battles_url": (meta or {}).get("source_battles_url"),
        }
    }
    return record


def infer_ms_per_tick(record: dict, plays: list[dict]) -> float:
    """
    Infer how many milliseconds correspond to one replay tick.

    Uses:
      duration_seconds (from timeline end marker like 5:10) and max play tick.

    Returns:
      ms_per_tick (float). Falls back to 50ms if we can't infer.
    """
    duration_s = record.get("duration_seconds")
    if not duration_s:
        return 50.0  # safe fallback (your data suggests ~52ms)

    ts = [e.get("t") for e in plays if isinstance(e.get("t"), int)]
    if not ts:
        return 50.0

    max_t = max(ts)
    if max_t <= 0:
        return 50.0

    return (float(duration_s) * 1000.0) / float(max_t)


# -----------------------------
# BC sample building (includes NOOP)
# -----------------------------
def build_bc_samples(record: dict, history_len: int = 20, tick_ms: int = 500) -> list[dict]:
    """
    Build behavior cloning samples at a fixed decision interval in MILLISECONDS.

    We convert milliseconds -> replay tick units using the replay's own duration.
    This avoids guessing whether e["t"] is ms or ticks.

    Labeling rule (per window):
      For each window [tick, tick + tick_dt_ticks):
        - If TEAM plays a card in that window, label = first TEAM card they played
        - Else label = "NOOP"

    History:
      - last `history_len` plays from both players BEFORE the window
      - each entry: {"p": "me"/"opp", "card": <card_id>, "t": <tick>}
    """
    events = record.get("events") or []

    team_deck = ((record.get("meta") or {}).get("team_deck")) or []
    opp_deck  = ((record.get("meta") or {}).get("opponent_deck")) or []

    # Infer which color corresponds to "team" (the player we scraped)
    place_events = [e for e in events if e.get("type") == "place"]
    inferred_team_side = infer_team_side_from_place_events(place_events)
    team_side = inferred_team_side or "blue"
    opp_side = "red" if team_side == "blue" else "blue"

    # Only use play events for actions
    plays = [e for e in events if e.get("type") == "play" and e.get("side") in ("blue", "red")]
    if not plays:
        return []

    plays.sort(key=lambda e: e["t"])

    # ---- ms -> tick conversion ----
    ms_per_tick = infer_ms_per_tick(record, plays)
    tick_dt = max(1, int(round(float(tick_ms) / ms_per_tick)))  # decision window size in tick units

    # Determine time range in replay ticks
    min_t = min(e["t"] for e in plays if isinstance(e.get("t"), int))
    max_t = max(e["t"] for e in plays if isinstance(e.get("t"), int))

    # Start at 0 so early-game NOOP exists (you can change to min_t if you want)
    tick = 0

    # Bin plays by window start tick
    plays_by_win: dict[int, list[dict]] = {}
    for e in plays:
        t = e["t"]
        win = (t // tick_dt) * tick_dt
        plays_by_win.setdefault(win, []).append(e)

    # Rolling history of plays (both players)
    history: list[dict] = []

    samples: list[dict] = []

    while tick <= max_t:
        window_plays = plays_by_win.get(tick, [])

        # Find first TEAM play in this window for label
        team_play = None
        for e in window_plays:
            if e.get("side") == team_side:
                team_play = e
                break

        label = team_play["card"] if team_play else "NOOP"

        # Sample uses history BEFORE applying this window's plays
        samples.append({
            "replay_id": record.get("replay_id"),
            "t": tick,                      # replay tick units
            "tick_ms": int(tick_ms),        # what you requested
            "tick_dt": int(tick_dt),        # window size in tick units (derived)
            "ms_per_tick": float(ms_per_tick),
            "player": "team",
            "team_side": team_side,
            "deck": team_deck,
            "opp_deck": opp_deck,
            "history": history[-history_len:],
            "label": label,
        })

        # Update history with all plays in this window (chronological)
        for e in window_plays:
            side = e.get("side")
            p = "me" if side == team_side else "opp"
            history.append({"p": p, "card": e.get("card"), "t": e.get("t")})

        tick += tick_dt

    return samples


# -----------------------------
# IO / CLI
# -----------------------------
def write_jsonl_lines(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_single_json(out_obj: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_directory(input_dir: Path, output_dir: Path, history_len: int, tick_ms: int, write_csv: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_files = sorted(input_dir.rglob("replay_*.json"))

    if not replay_files:
        print("No replay JSON files found.")
        return

    print(f"Found {len(replay_files)} replay files")

    for replay_path in replay_files:
        if replay_path.name == "manifest.json":
            continue

        parsed = parse_replay_json_file(replay_path)

        # 1) replay summary (single json object, .jsonl name kept for compatibility if you want)
        summary_path = output_dir / f"replay_{parsed['replay_id']}.jsonl"
        write_jsonl_lines([parsed], summary_path)

        # 2) bc samples
        samples = build_bc_samples(parsed, history_len=history_len, tick_ms=tick_ms)
        bc_path = output_dir / f"bc_{parsed['replay_id']}.jsonl"
        write_jsonl_lines(samples, bc_path)

        # optional: flat CSV of events
        if write_csv:
            csv_path = output_dir / f"events_{parsed['replay_id']}.csv"
            write_events_csv(parsed.get("events") or [], csv_path)

        print(f"Parsed → {summary_path.name} | BC → {bc_path.name} ({len(samples)} samples)")


def write_events_csv(events: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for e in events for k in e.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in events:
            w.writerow(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing raw replay JSON files")
    ap.add_argument("--output_dir", required=True, help="Directory to save parsed BC-ready files")
    ap.add_argument("--history_len", type=int, default=20, help="How many past actions to include in the model input")
    ap.add_argument("--tick_ms", type=int, default=250, help="Decision tick size in milliseconds (controls NOOP frequency)")
    ap.add_argument("--csv", action="store_true", help="Also emit events_<id>.csv for debugging")
    args = ap.parse_args()

    parse_directory(Path(args.input_dir), Path(args.output_dir), args.history_len, args.tick_ms, write_csv=bool(args.csv))


if __name__ == "__main__":
    main()
