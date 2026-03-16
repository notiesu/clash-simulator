import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset

# =========================
# Special tokens (keep consistent with your tokenizer / training)
# =========================
PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
NOOP = "NOOP"


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _to_int_or_none(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        if v.lower() == "none":
            return None
        try:
            return int(float(v))
        except Exception:
            return None
    return None


# =========================
# HTML replay coordinate system (from your provided replay_map markers)
# data-x in [500 .. 17500]
# data-y in [500 .. 31499]
# =========================
HTML_X_MIN = 500
HTML_X_MAX = 17500
HTML_Y_MIN = 500
HTML_Y_MAX = 31499

# Env tile grid
X_BINS = 18  # x_tile: 0..17
Y_BINS = 32  # y_tile: 0..31


def _coords_to_tiles(
    x_raw: Optional[int],
    y_raw: Optional[int],
    x_bins: int = X_BINS,
    y_bins: int = Y_BINS,
    x_min: int = HTML_X_MIN,
    x_max: int = HTML_X_MAX,
    y_min: int = HTML_Y_MIN,
    y_max: int = HTML_Y_MAX,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Convert raw HTML replay coords (data-x/data-y) into discrete env tile bins:
      x_tile in [0..17] (18 bins)
      y_tile in [0..31] (32 bins)
    """
    if x_raw is None or y_raw is None:
        return None, None
    if x_max <= x_min or y_max <= y_min:
        return None, None

    fx = (x_raw - x_min) / (x_max - x_min)
    fy = (y_raw - y_min) / (y_max - y_min)

    # floor binning, then clamp
    x_tile = int(fx * x_bins)
    y_tile = int(fy * y_bins)

    x_tile = _clamp_int(x_tile, 0, x_bins - 1)
    y_tile = _clamp_int(y_tile, 0, y_bins - 1)
    return x_tile, y_tile


def _extract_xy_tile(obj: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    Try to extract placement (x_tile, y_tile) from a dict.
    Accepts:
      - x_tile / y_tile (already discrete)
      - x / y (raw html coords), binned to tiles
      - nested pos/position/tile dict
    """
    xt = _to_int_or_none(obj.get("x_tile"))
    yt = _to_int_or_none(obj.get("y_tile"))
    if xt is not None and yt is not None:
        return _clamp_int(xt, 0, 17), _clamp_int(yt, 0, 31)

    x_raw = _to_int_or_none(obj.get("x"))
    y_raw = _to_int_or_none(obj.get("y"))
    if x_raw is not None and y_raw is not None:
        return _coords_to_tiles(x_raw, y_raw)

    pos = obj.get("pos") or obj.get("position") or obj.get("tile")
    if isinstance(pos, dict):
        xt = _to_int_or_none(pos.get("x_tile"))
        yt = _to_int_or_none(pos.get("y_tile"))
        if xt is not None and yt is not None:
            return _clamp_int(xt, 0, 17), _clamp_int(yt, 0, 31)

        x_raw = _to_int_or_none(pos.get("x"))
        y_raw = _to_int_or_none(pos.get("y"))
        if x_raw is not None and y_raw is not None:
            return _coords_to_tiles(x_raw, y_raw)

    return None, None


class BehaviorCloningDataset(Dataset):
    """
    Behavior Cloning Dataset for Option 1 (factorized heads):
      - gate_y: 0 WAIT, 1 PLAY
      - y:      0..7 deck slot i, 8 NOOP
      - target_x: 0..17 (only meaningful when gate_y=1)
      - target_y: 0..31 (only meaningful when gate_y=1)

    Inputs:
      - history_cards:   (H,) token ids
      - history_players: (H,) 0=team, 1=opp
      - history_x:       (H,) 0..17 or PAD_X (=18)
      - history_y:       (H,) 0..31 or PAD_Y (=32)
      - deck, opp_deck:  (8,) token ids

    JSONL row expected shape (flexible):
      {
        "history": [{"card": "...", "p":"team"/"opp", "x":int, "y":int} ...],
        "deck": [...8...],
        "opp_deck": [...8...],
        "label": "<card-name>" OR "NOOP",
        "x": int, "y": int   # label placement (optional)
      }
    """

    # Must match model.py pad_x/pad_y
    PAD_X = 18  # x: 0..17 valid
    PAD_Y = 32  # y: 0..31 valid

    def __init__(self, jsonl_path: Path, history_len: int = 20):
        self.jsonl_path = Path(jsonl_path)
        self.history_len = history_len

        # Load JSONL rows
        self.rows: List[Dict[str, Any]] = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

        # Build vocab (cards only)
        vocab = {PAD, UNK, BOS, EOS, NOOP}
        for r in self.rows:
            for h in r.get("history", []):
                c = h.get("card")
                if c:
                    vocab.add(c)
            for c in r.get("deck", []):
                if c:
                    vocab.add(c)
            for c in r.get("opp_deck", []):
                if c:
                    vocab.add(c)
            lbl = r.get("label")
            if lbl:
                vocab.add(lbl)

        self.token2id = {tok: i for i, tok in enumerate(sorted(vocab))}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

        self.pad_id = self.token2id[PAD]
        self.unk_id = self.token2id[UNK]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.rows[idx]

        # -------------------------
        # 1) HISTORY
        # -------------------------
        history = sample.get("history", [])

        cards: List[int] = []
        players: List[int] = []
        xs: List[int] = []
        ys: List[int] = []

        for h in history[-self.history_len :]:
            card = h.get("card", UNK)
            player = h.get("p", "team")  # "team" or "opp"

            x_tile, y_tile = _extract_xy_tile(h)

            cards.append(self.token2id.get(card, self.unk_id))
            players.append(0 if player == "team" else 1)

            xs.append(self.PAD_X if x_tile is None else _clamp_int(x_tile, 0, 17))
            ys.append(self.PAD_Y if y_tile is None else _clamp_int(y_tile, 0, 31))

        # Left-pad to fixed history_len
        pad_n = self.history_len - len(cards)
        if pad_n > 0:
            cards = [self.pad_id] * pad_n + cards
            players = [0] * pad_n + players
            xs = [self.PAD_X] * pad_n + xs
            ys = [self.PAD_Y] * pad_n + ys

        history_cards = torch.tensor(cards, dtype=torch.long)
        history_players = torch.tensor(players, dtype=torch.long)
        history_x = torch.tensor(xs, dtype=torch.long)
        history_y = torch.tensor(ys, dtype=torch.long)

        # -------------------------
        # 2) DECKS (slot order preserved)
        # -------------------------
        deck = [c for c in sample.get("deck", []) if c]
        if len(deck) != 8:
            deck = (deck + [UNK] * 8)[:8]

        opp_deck = [c for c in sample.get("opp_deck", []) if c]
        if len(opp_deck) != 8:
            opp_deck = (opp_deck + [UNK] * 8)[:8]

        deck_ids = torch.tensor([self.token2id.get(c, self.unk_id) for c in deck], dtype=torch.long)
        opp_deck_ids = torch.tensor([self.token2id.get(c, self.unk_id) for c in opp_deck], dtype=torch.long)

        # -------------------------
        # 3) ACTION LABELS
        # -------------------------
        action_list = deck + [NOOP]  # slot 0..7 + NOOP

        label = sample.get("label", NOOP)
        if label not in action_list:
            label = NOOP
        y = action_list.index(label)  # 0..8

        gate_y = 0 if y == 8 else 1  # WAIT if NOOP else PLAY

        # Label placement (top-level x/y or x_tile/y_tile)
        x_tile, y_tile = _extract_xy_tile(sample)

        # Some pipelines store label placement nested
        if (x_tile is None) or (y_tile is None):
            lbl_pos = sample.get("label_pos") or sample.get("label_position")
            if isinstance(lbl_pos, dict):
                x2, y2 = _extract_xy_tile(lbl_pos)
                if x2 is not None and y2 is not None:
                    x_tile, y_tile = x2, y2

        # Default dummies if missing (we'll mask losses in train.py)
        if x_tile is None:
            x_tile = 0
        if y_tile is None:
            y_tile = 0

        target_x = _clamp_int(x_tile, 0, 17)
        target_y = _clamp_int(y_tile, 0, 31)

        return {
            "history_cards": history_cards,
            "history_players": history_players,
            "history_x": history_x,
            "history_y": history_y,
            "deck": deck_ids,
            "opp_deck": opp_deck_ids,
            "gate_y": torch.tensor(gate_y, dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.long),
            "target_x": torch.tensor(target_x, dtype=torch.long),
            "target_y": torch.tensor(target_y, dtype=torch.long),
        }


def collate_pad(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "history_cards": torch.stack([b["history_cards"] for b in batch]),
        "history_players": torch.stack([b["history_players"] for b in batch]),
        "history_x": torch.stack([b["history_x"] for b in batch]),
        "history_y": torch.stack([b["history_y"] for b in batch]),
        "deck": torch.stack([b["deck"] for b in batch]),
        "opp_deck": torch.stack([b["opp_deck"] for b in batch]),
        "gate_y": torch.stack([b["gate_y"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "target_x": torch.stack([b["target_x"] for b in batch]),
        "target_y": torch.stack([b["target_y"] for b in batch]),
    }
