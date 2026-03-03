#!/usr/bin/env python3
"""
Flip RoyaleAPI replay JSONs so Hog 2.6 is always the "team" (blue / bottom perspective).

DEFAULT behavior ASSUMES your env's X is mirrored relative to replay/html:
  env (0,0) <-> replay (18,0)
  env (18,0) <-> replay (0,0)

So a 180° rotation in env-space becomes (in replay/html space):
  - Y flips
  - X unchanged   <-- DEFAULT

If you ever need the old behavior (flip both X and Y in html space), pass:
  --no_env_x_is_already_flipped
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HTML_X_MIN = 500
HTML_X_MAX = 17500
HTML_Y_MIN = 500
HTML_Y_MAX = 31499

HOG26 = [
    "cannon",
    "fireball",
    "hog-rider",
    "ice-golem",
    "ice-spirit",
    "musketeer",
    "skeletons",
    "the-log",
]
HOG26_SET = frozenset(HOG26)


@dataclass(frozen=True)
class Bounds:
    xmin: int
    xmax: int
    ymin: int
    ymax: int

    def flip_x(self, x: int) -> int:
        return self.xmin + (self.xmax - x)

    def flip_y(self, y: int) -> int:
        return self.ymin + (self.ymax - y)

    def in_x(self, x: int) -> bool:
        return self.xmin <= x <= self.xmax

    def in_y(self, y: int) -> bool:
        return self.ymin <= y <= self.ymax


BOUNDS = Bounds(HTML_X_MIN, HTML_X_MAX, HTML_Y_MIN, HTML_Y_MAX)


def norm_card_name(s: str) -> str:
    return s.strip().lower()


def deck_is_hog26(deck: Any) -> bool:
    if not isinstance(deck, list) or len(deck) != 8:
        return False
    deck_set = frozenset(norm_card_name(x) for x in deck if isinstance(x, str))
    return deck_set == HOG26_SET


def find_hog26_side(meta: dict[str, Any]) -> str | None:
    team_decks = meta.get("team_decks") or []
    opp_decks = meta.get("opponent_decks") or []

    for d in team_decks:
        if deck_is_hog26(d):
            return "team"
    for d in opp_decks:
        if deck_is_hog26(d):
            return "opponent"
    return None


def swap_meta_team_opponent(meta: dict[str, Any]) -> None:
    swap_pairs = [
        ("team_tags", "opponent_tags"),
        ("team_decks", "opponent_decks"),
    ]
    for a, b in swap_pairs:
        if a in meta or b in meta:
            meta[a], meta[b] = meta.get(b), meta.get(a)


_BLUE = "__SWAP_BLUE__"
_RED = "__SWAP_RED__"


def swap_blue_red_in_html(html: str) -> str:
    # quick token swap; placeholders avoid double swap
    html = re.sub(r"\bblue\b", _BLUE, html)
    html = re.sub(r"\bred\b", _RED, html)
    return html.replace(_BLUE, "red").replace(_RED, "blue")


# Match data-x="123" OR data-x='123' (after json.loads, quotes are real quotes)
_RE_DATA_X = re.compile(r'data-x=(?P<q>["\'])(?P<v>\d+)(?P=q)')
_RE_DATA_Y = re.compile(r'data-y=(?P<q>["\'])(?P<v>\d+)(?P=q)')


def flip_xy_in_html_markers(html: str, bounds: Bounds, *, env_x_already_flipped: bool) -> str:
    """
    If env_x_already_flipped is True (DEFAULT):
      - X unchanged
      - Y flipped
    """

    def repl_x(m: re.Match) -> str:
        q = m.group("q")
        x = int(m.group("v"))
        if bounds.in_x(x) and (not env_x_already_flipped):
            x = bounds.flip_x(x)
        return f'data-x={q}{x}{q}'

    def repl_y(m: re.Match) -> str:
        q = m.group("q")
        y = int(m.group("v"))
        if bounds.in_y(y):
            y = bounds.flip_y(y)
        return f'data-y={q}{y}{q}'

    html = _RE_DATA_X.sub(repl_x, html)
    html = _RE_DATA_Y.sub(repl_y, html)
    return html


def flip_xy_in_json(obj: Any, bounds: Bounds, *, env_x_already_flipped: bool) -> Any:
    """
    Recursively flip dict keys exactly named "x" and "y" if they are ints within bounds.

    DEFAULT (env_x_already_flipped=True):
      - X unchanged
      - Y flipped
    """
    if isinstance(obj, list):
        return [flip_xy_in_json(v, bounds, env_x_already_flipped=env_x_already_flipped) for v in obj]
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            nv = flip_xy_in_json(v, bounds, env_x_already_flipped=env_x_already_flipped)

            if k == "x" and isinstance(nv, int) and bounds.in_x(nv):
                if not env_x_already_flipped:
                    nv = bounds.flip_x(nv)
            elif k == "y" and isinstance(nv, int) and bounds.in_y(nv):
                nv = bounds.flip_y(nv)

            out[k] = nv
        return out
    return obj


def process_replay_json(
    d: dict[str, Any],
    bounds: Bounds,
    *,
    env_x_already_flipped: bool,
) -> tuple[dict[str, Any], bool]:
    meta = d.get("meta")
    data = d.get("data")
    if not isinstance(meta, dict) or not isinstance(data, dict):
        return d, False

    side = find_hog26_side(meta)
    if side != "opponent":
        return d, False

    swap_meta_team_opponent(meta)

    d = flip_xy_in_json(d, bounds, env_x_already_flipped=env_x_already_flipped)

    html = d.get("data", {}).get("html")
    if isinstance(html, str):
        html = swap_blue_red_in_html(html)
        html = flip_xy_in_html_markers(html, bounds, env_x_already_flipped=env_x_already_flipped)
        d["data"]["html"] = html

    return d, True


def iter_json_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".json":
            yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument(
        "--no_env_x_is_already_flipped",
        action="store_true",
        help="If set: treat env X as NOT mirrored vs replay/html (old behavior flips X and Y).",
    )

    args = ap.parse_args()
    env_x_already_flipped = not args.no_env_x_is_already_flipped

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir

    if not in_dir.exists():
        raise SystemExit(f"in_dir does not exist: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = n_changed = n_copied = n_failed = n_skipped = 0

    for src in iter_json_files(in_dir):
        n_total += 1
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not args.overwrite:
            n_skipped += 1
            continue

        try:
            raw = src.read_text(encoding="utf-8")
            d = json.loads(raw)

            if not isinstance(d, dict):
                shutil.copy2(src, dst)
                n_copied += 1
                continue

            new_d, changed = process_replay_json(
                d,
                BOUNDS,
                env_x_already_flipped=env_x_already_flipped,
            )

            if changed:
                n_changed += 1
                dst.write_text(json.dumps(new_d, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                shutil.copy2(src, dst)
                n_copied += 1

        except Exception:
            try:
                shutil.copy2(src, dst)
                n_failed += 1
            except Exception:
                n_failed += 1

    print("Done.")
    print(f"  env_x_already_flipped (DEFAULT): {env_x_already_flipped}")
    print(f"  total json files seen: {n_total}")
    print(f"  changed (hog was opponent): {n_changed}")
    print(f"  copied as-is: {n_copied}")
    print(f"  skipped (exists, no --overwrite): {n_skipped}")
    print(f"  failed but copied original: {n_failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())