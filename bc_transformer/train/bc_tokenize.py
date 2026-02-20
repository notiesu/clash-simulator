"""
bc_tokenize.py

Tokenize RoyaleAPI replay HTML markers into bc_*.jsonl rows that your updated
datasets.py + model.py + train.py (Option 1) can consume.

What it produces (per row):
{
  "replay_id": "...",
  "history": [
     {"card": "giant", "p": "opp",  "x": 9500, "y": 31499, "t": 1253},
     {"card": "cannon","p": "team", "x": 8500, "y": 9500,  "t": 582},
     ...
  ],
  "deck": [...8 cards...],
  "opp_deck": [...8 cards...],
  "label": "ice-spirit" OR "NOOP",
  "x": 9500,     # only when label != NOOP
  "y": 10500,    # only when label != NOOP
  "t": 1622
}

Notes:
- We do NOT guess coord ranges here; we just store raw (data-x, data-y).
  Your datasets.py will map raw coords to (x_tile,y_tile) using:
    x in [500..17500], y in [500..31499]
- We optionally generate NOOP (WAIT) rows after opponent actions to train the gate.
  This helps the gate not become useless.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Token constants (MUST stay consistent across repo)
# ============================================================
PAD = "<PAD>"
BOS = "<BOS>"
ME = "<ME>"
OPP = "<OPP>"

# Action constant used in data (and slot-policy action_list)
NOOP = "NOOP"


# ============================================================
# Optional helper: time-delta bucketing
# (You can use this later if you want the model to see timing)
# ============================================================
def bucket_time_delta(dt, n_buckets: int = 16, dt_max: float = 10.0) -> int:
    """
    Log-ish bucketing so tiny deltas are separated better.
    dt in seconds; cap at dt_max.
    returns int bucket in [0, n_buckets-1]
    """
    dt = max(0.0, min(float(dt), float(dt_max)))
    x = dt / dt_max if dt_max > 0 else 0.0
    y = math.log1p(9 * x) / math.log1p(9)  # 0..1
    b = int(y * (n_buckets - 1) + 1e-9)
    return max(0, min(n_buckets - 1, b))


# ============================================================
# Parsing markers from replay HTML
# ============================================================

# Matches marker divs that contain the replay placement info.
# Example chunk you pasted:
# <div class="red marker" data-x="8499" data-y="31499" data-c="bomber" data-t="208" ... data-s="o" ...>
_MARKER_RE = re.compile(
    r'<div\s+class="[^"]*\bmarker\b[^"]*"\s+'
    r'[^>]*\bdata-x="(?P<x>[^"]*)"\s+'
    r'[^>]*\bdata-y="(?P<y>[^"]*)"\s+'
    r'[^>]*\bdata-c="(?P<c>[^"]*)"\s+'
    r'[^>]*\bdata-t="(?P<t>\d+)"\s+'
    r'[^>]*\bdata-i="(?P<i>\d+)"\s+'
    r'[^>]*\bdata-s="(?P<s>[to])"\s*'
    r'[^>]*>',
    re.IGNORECASE,
)


def _to_int_or_none(v: str) -> Optional[int]:
    if v is None:
        return None
    v = str(v).strip()
    if v.lower() == "none":
        return None
    try:
        return int(v)
    except Exception:
        return None


@dataclass(frozen=True)
class Marker:
    t: int              # time tick from data-t
    side: str           # 't' team, 'o' opponent
    card: str           # data-c
    x: Optional[int]    # data-x (raw html coord)
    y: Optional[int]    # data-y (raw html coord)


def parse_markers_from_html(html: str) -> List[Marker]:
    markers: List[Marker] = []
    for m in _MARKER_RE.finditer(html):
        x = _to_int_or_none(m.group("x"))
        y = _to_int_or_none(m.group("y"))
        card = (m.group("c") or "").strip()
        t = int(m.group("t"))
        side = (m.group("s") or "").strip().lower()

        # Skip invalid markers
        if not card or card == "_invalid":
            continue

        markers.append(Marker(t=t, side=side, card=card, x=x, y=y))

    # Sort by time, stable
    markers.sort(key=lambda z: z.t)
    return markers


# ============================================================
# Row building
# ============================================================

def _get_decks(meta: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    meta['team_decks'] and meta['opponent_decks'] are typically:
      [[<8 cards>], [<8 cards>], ...]
    We'll take the first deck by default.
    """
    team_decks = meta.get("team_decks") or []
    opp_decks = meta.get("opponent_decks") or []

    deck = []
    opp_deck = []

    if isinstance(team_decks, list) and team_decks:
        if isinstance(team_decks[0], list):
            deck = [str(c) for c in team_decks[0]]
    if isinstance(opp_decks, list) and opp_decks:
        if isinstance(opp_decks[0], list):
            opp_deck = [str(c) for c in opp_decks[0]]

    # Ensure length 8 if possible (leave as-is if input is weird)
    if len(deck) > 8:
        deck = deck[:8]
    if len(opp_deck) > 8:
        opp_deck = opp_deck[:8]

    return deck, opp_deck


def _push_history(history: List[Dict[str, Any]], event: Dict[str, Any], history_len: int) -> None:
    history.append(event)
    if len(history) > history_len:
        del history[0 : len(history) - history_len]


def build_bc_rows_from_replay(
    replay_json: Dict[str, Any],
    history_len: int = 20,
    add_noop_after_opp: bool = True,
    noop_gap_ticks: int = 120,
    max_noop_rows_per_replay: int = 200,
) -> List[Dict[str, Any]]:
    """
    Creates rows mostly at TEAM action times, plus optional NOOP rows after OPP actions.

    add_noop_after_opp:
      After an opponent action at time t_opp, if the next team action occurs at time t_team
      and (t_team - t_opp) >= noop_gap_ticks, then emit a NOOP row at time t_opp
      (history contains the state *including* that opponent action).
    """
    meta = replay_json.get("meta", {})
    data = replay_json.get("data", {})
    html = (data.get("html") or "")

    replay_id = meta.get("replay_id") or meta.get("id") or "unknown_replay"
    deck, opp_deck = _get_decks(meta)

    markers = parse_markers_from_html(html)

    # Precompute next TEAM action time after each index (for NOOP logic)
    next_team_time_after: List[Optional[int]] = [None] * len(markers)
    next_t = None
    for i in range(len(markers) - 1, -1, -1):
        next_team_time_after[i] = next_t
        if markers[i].side == "t":
            next_t = markers[i].t

    rows: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []

    noop_count = 0

    for i, mk in enumerate(markers):
        # Build event dict for history
        p = "team" if mk.side == "t" else "opp"
        event = {"card": mk.card, "p": p, "t": mk.t}

        # Only include x/y if they exist (some spells/invalids might not)
        if mk.x is not None and mk.y is not None:
            event["x"] = mk.x
            event["y"] = mk.y

        # Update history with the event
        _push_history(history, event, history_len)

        # Optional NOOP row after opponent actions (to train gate WAIT)
        if add_noop_after_opp and mk.side == "o" and noop_count < max_noop_rows_per_replay:
            next_team_t = next_team_time_after[i]
            if next_team_t is None:
                gap_ok = True  # no more team actions; waiting is valid
            else:
                gap_ok = (next_team_t - mk.t) >= int(noop_gap_ticks)

            if gap_ok:
                rows.append(
                    {
                        "replay_id": replay_id,
                        "t": mk.t,
                        "deck": deck,
                        "opp_deck": opp_deck,
                        "history": list(history),  # copy
                        "label": NOOP,
                        # NOOP has no x/y placement
                    }
                )
                noop_count += 1

        # TEAM action row (this is what we primarily learn)
        if mk.side == "t":
            # If placement missing, we still emit the row; train.py will mask x/y if missing
            row = {
                "replay_id": replay_id,
                "t": mk.t,
                "deck": deck,
                "opp_deck": opp_deck,
                "history": list(history[:-1]),  # history BEFORE the label action
                "label": mk.card,
            }
            if mk.x is not None and mk.y is not None:
                row["x"] = mk.x
                row["y"] = mk.y
            rows.append(row)

    return rows


# ============================================================
# IO / CLI
# ============================================================

def load_replay_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_replay_files(inp: Path) -> List[Path]:
    """
    Recursively find replay json files.

    - If inp is a file: return [inp]
    - If inp is a dir: recursively collect replay_*.json (preferred),
      else fall back to any *.json.
    """
    inp = Path(inp)
    if inp.is_file():
        return [inp]

    if inp.is_dir():
        files = sorted(inp.rglob("replay_*.json"))
        if not files:
            files = sorted(inp.rglob("*.json"))
        return files

    raise FileNotFoundError(f"Input not found: {inp}")


def write_jsonl(rows, out_path: Path) -> int:
    """
    Stream-write JSONL rows to disk. Returns number of rows written.
    This avoids storing the full dataset in RAM.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="replay_*.json file OR directory of replay json files")
    ap.add_argument("--output", type=Path, required=True, help="Output .jsonl path (recommended name: bc_*.jsonl)")
    ap.add_argument("--history_len", type=int, default=20)
    ap.add_argument("--add_noop_after_opp", action="store_true", help="Emit NOOP rows after opponent actions")
    ap.add_argument("--noop_gap_ticks", type=int, default=120, help="If next team action is >= this many ticks away, emit NOOP after opp action")
    ap.add_argument("--max_noop_rows_per_replay", type=int, default=200)

    args = ap.parse_args()

    replay_files = iter_replay_files(args.input)
    if not replay_files:
        raise FileNotFoundError(f"No replay json files found at: {args.input}")

    def row_iter():
        # Generator that yields rows across all replay files
        for rp in replay_files:
            rj = load_replay_json(rp)
            rows = build_bc_rows_from_replay(
                rj,
                history_len=args.history_len,
                add_noop_after_opp=bool(args.add_noop_after_opp),
                noop_gap_ticks=int(args.noop_gap_ticks),
                max_noop_rows_per_replay=int(args.max_noop_rows_per_replay),
            )
            # rows is a list per replay; we stream them out immediately
            for r in rows:
                yield r

    n = write_jsonl(row_iter(), args.output)
    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
