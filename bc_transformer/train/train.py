# train.py
from __future__ import annotations

import os
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from model import BCTransformer
from bc_tokenize import build_bc_rows_from_replay

PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
NOOP = "NOOP"

# HTML replay coordinate system (from RoyaleAPI replay_map markers)
# data-x in [500 .. 17500]
# data-y in [500 .. 31499]
HTML_X_MIN = 500
HTML_X_MAX = 17500
HTML_Y_MIN = 500
HTML_Y_MAX = 31499

# Env tile grid
X_BINS = 18  # x_tile: 0..17
Y_BINS = 32  # y_tile: 0..31

# Must match model.py pad_x/pad_y
PAD_X = 18
PAD_Y = 32


def list_or_build_jsonl(p: Path, auto_tokenize: bool = True) -> List[Path]:
    """
    If p contains bc_*.jsonl → use them.
    If p contains replay_*.json and auto_tokenize=True → build bc_auto.jsonl (STREAMING).
    """
    p = Path(p)

    if p.is_file():
        return [p]

    if not p.is_dir():
        raise FileNotFoundError(f"{p} is not valid")

    # 1) Check for existing bc_*.jsonl
    jsonl_files = sorted(p.rglob("bc_*.jsonl"))
    if jsonl_files:
        print(f"Found {len(jsonl_files)} bc_*.jsonl files.")
        return jsonl_files

    # 2) Otherwise look for replay JSONs
    replay_files = sorted(p.rglob("replay_*.json"))
    if not replay_files:
        raise FileNotFoundError(f"No bc_*.jsonl or replay_*.json found under {p}")

    if not auto_tokenize:
        raise RuntimeError("Replay JSONs found but auto_tokenize disabled")

    print(f"Found {len(replay_files)} replay files. Auto-tokenizing (streaming)...")

    out_path = p / "bc_auto.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for rp in replay_files:
            with open(rp, "r", encoding="utf-8") as f:
                replay_json = json.load(f)

            # NOTE: build_bc_rows_from_replay returns a list per replay.
            # That's usually fine; the big OOM was collecting *all replays* into one list.
            rows = build_bc_rows_from_replay(replay_json)
            for r in rows:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1

    print(f"Auto-generated {n} rows → {out_path}")
    return [out_path]


def iter_jsonl_rows(files: List[Path]) -> Iterator[Dict[str, Any]]:
    """Stream JSONL rows from one or more files."""
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def build_vocab_from_jsonl_files(train_files: List[Path]) -> Dict[str, int]:
    """Build token2id from train files only (streaming, no giant list in RAM)."""
    vocab: Set[str] = {PAD, UNK, BOS, EOS, NOOP}

    for r in iter_jsonl_rows(train_files):
        # history cards
        hist = r.get("history", [])
        if isinstance(hist, list):
            for h in hist:
                if isinstance(h, dict):
                    c = h.get("card")
                    if isinstance(c, str) and c:
                        vocab.add(c)

        # decks
        for k in ("deck", "opp_deck"):
            d = r.get(k, [])
            if isinstance(d, list):
                for c in d:
                    if isinstance(c, str) and c:
                        vocab.add(c)

        # label
        lbl = r.get("label")
        if isinstance(lbl, str) and lbl:
            vocab.add(lbl)

    return {tok: i for i, tok in enumerate(sorted(vocab))}


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


def _coords_to_tiles(x_raw: Optional[int], y_raw: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    """
    Convert raw HTML coords (data-x/data-y) into discrete env tile bins:
      x_tile in [0..17] (18 bins)
      y_tile in [0..31] (32 bins)
    """
    if x_raw is None or y_raw is None:
        return None, None

    fx = (x_raw - HTML_X_MIN) / (HTML_X_MAX - HTML_X_MIN)
    fy = (y_raw - HTML_Y_MIN) / (HTML_Y_MAX - HTML_Y_MIN)

    x_tile = int(fx * X_BINS)
    y_tile = int(fy * Y_BINS)

    x_tile = _clamp_int(x_tile, 0, X_BINS - 1)
    y_tile = _clamp_int(y_tile, 0, Y_BINS - 1)
    return x_tile, y_tile


def _extract_xy_tile(obj: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    Try to extract placement (x_tile, y_tile) from a dict.
    Accepts:
      - x_tile / y_tile (already discrete)
      - x / y raw coords, mapped to tiles
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


class BCJsonlIterableDataset(IterableDataset):
    """
    Stream rows from JSONL and tensorize on-the-fly (constant-memory dataset).
    If task=='card', filters to rows where label is a real play AND is in deck.
    """

    def __init__(
        self,
        files: List[Path],
        history_len: int,
        token2id: Dict[str, int],
        pad_id: int,
        unk_id: int,
        task: str,
        shuffle_buffer: int = 0,
    ):
        super().__init__()
        self.files = files
        self.history_len = int(history_len)
        self.token2id = token2id
        self.pad_id = int(pad_id)
        self.unk_id = int(unk_id)
        self.task = task
        self.shuffle_buffer = int(shuffle_buffer)

    def _is_good_play_row(self, r: Dict[str, Any]) -> bool:
        lbl = r.get("label", None)
        if not (isinstance(lbl, str) and lbl and lbl != NOOP):
            return False
        deck = r.get("deck", [])
        return isinstance(deck, list) and (lbl in deck)

    def _tensorize(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # ---------------------------
        # HISTORY (fixed length)
        # ---------------------------
        history = sample.get("history", [])
        cards: List[int] = []
        players: List[int] = []
        xs: List[int] = []
        ys: List[int] = []

        if isinstance(history, list):
            for h in history:
                if not isinstance(h, dict):
                    continue
                card = h.get("card", UNK)
                p = h.get("p", "team")
                cards.append(self.token2id.get(card, self.unk_id))
                players.append(0 if p == "team" else 1)

                x_tile, y_tile = _extract_xy_tile(h)
                xs.append(PAD_X if x_tile is None else _clamp_int(x_tile, 0, 17))
                ys.append(PAD_Y if y_tile is None else _clamp_int(y_tile, 0, 31))

        # keep last history_len
        cards = cards[-self.history_len :]
        players = players[-self.history_len :]
        xs = xs[-self.history_len :]
        ys = ys[-self.history_len :]

        # make all same length (in case any list got out of sync)
        m = min(len(cards), len(players), len(xs), len(ys))
        cards, players, xs, ys = cards[-m:], players[-m:], xs[-m:], ys[-m:]

        # left-pad
        pad_n = self.history_len - len(cards)
        if pad_n > 0:
            cards = [self.pad_id] * pad_n + cards
            players = [0] * pad_n + players
            xs = [PAD_X] * pad_n + xs
            ys = [PAD_Y] * pad_n + ys

        history_cards = torch.tensor(cards, dtype=torch.long)
        history_players = torch.tensor(players, dtype=torch.long)
        history_x = torch.tensor(xs, dtype=torch.long)
        history_y = torch.tensor(ys, dtype=torch.long)

        # ---------------------------
        # DECKS (preserve order, force len 8)
        # ---------------------------
        deck = sample.get("deck", [])
        if not isinstance(deck, list):
            deck = []
        deck = [c for c in deck if isinstance(c, str) and c]
        if len(deck) != 8:
            deck = (deck + [UNK] * 8)[:8]

        opp_deck = sample.get("opp_deck", [])
        if not isinstance(opp_deck, list):
            opp_deck = []
        opp_deck = [c for c in opp_deck if isinstance(c, str) and c]
        if len(opp_deck) != 8:
            opp_deck = (opp_deck + [UNK] * 8)[:8]

        deck_ids = torch.tensor([self.token2id.get(c, self.unk_id) for c in deck], dtype=torch.long)
        opp_deck_ids = torch.tensor([self.token2id.get(c, self.unk_id) for c in opp_deck], dtype=torch.long)

        # ---------------------------
        # LABELS: gate + card-slot
        # ---------------------------
        lbl = sample.get("label", NOOP)
        is_play = isinstance(lbl, str) and lbl and (lbl != NOOP)
        y_gate = torch.tensor(1 if is_play else 0, dtype=torch.long)  # 0=WAIT, 1=PLAY

        action_list = deck + [NOOP]
        label = lbl if isinstance(lbl, str) and lbl else NOOP
        if label not in action_list:
            label = NOOP
        y_card = torch.tensor(action_list.index(label), dtype=torch.long)  # 0..8

        # ---------------------------
        # LABELS: placement (x/y)
        # ---------------------------
        x_tile, y_tile = _extract_xy_tile(sample)
        if (x_tile is None) or (y_tile is None):
            lbl_pos = sample.get("label_pos") or sample.get("label_position")
            if isinstance(lbl_pos, dict):
                x2, y2 = _extract_xy_tile(lbl_pos)
                if x2 is not None and y2 is not None:
                    x_tile, y_tile = x2, y2

        has_xy = 1 if (x_tile is not None and y_tile is not None) else 0

        # Defaults (masked out when has_xy=0 or gate=0)
        if x_tile is None:
            x_tile = 0
        if y_tile is None:
            y_tile = 0

        target_x = torch.tensor(_clamp_int(x_tile, 0, 17), dtype=torch.long)
        target_y = torch.tensor(_clamp_int(y_tile, 0, 31), dtype=torch.long)
        has_xy_t = torch.tensor(has_xy, dtype=torch.long)

        return {
            "history_cards": history_cards,
            "history_players": history_players,
            "history_x": history_x,
            "history_y": history_y,
            "deck": deck_ids,
            "opp_deck": opp_deck_ids,
            "y_gate": y_gate,
            "y_card": y_card,
            "target_x": target_x,
            "target_y": target_y,
            "has_xy": has_xy_t,
        }

    def __iter__(self):
        # Optional bounded shuffle (keeps RAM bounded)
        if self.shuffle_buffer > 0:
            buf: List[Dict[str, Any]] = []
            for r in iter_jsonl_rows(self.files):
                if self.task == "card" and not self._is_good_play_row(r):
                    continue
                buf.append(r)
                if len(buf) >= self.shuffle_buffer:
                    random.shuffle(buf)
                    for item in buf:
                        yield self._tensorize(item)
                    buf.clear()
            if buf:
                random.shuffle(buf)
                for item in buf:
                    yield self._tensorize(item)
        else:
            for r in iter_jsonl_rows(self.files):
                if self.task == "card" and not self._is_good_play_row(r):
                    continue
                yield self._tensorize(r)


def collate_dynamic(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    def pad_1d(t: torch.Tensor, L: int, pad_value: int) -> torch.Tensor:
        if t.numel() == L:
            return t
        if t.numel() > L:
            return t[-L:]
        out = torch.full((L,), pad_value, dtype=t.dtype)
        out[-t.numel() :] = t
        return out

    max_h = max(b["history_cards"].numel() for b in batch)

    history_cards = torch.stack([pad_1d(b["history_cards"], max_h, pad_id) for b in batch])
    history_players = torch.stack([pad_1d(b["history_players"], max_h, 0) for b in batch])
    history_x = torch.stack([pad_1d(b["history_x"], max_h, PAD_X) for b in batch])
    history_y = torch.stack([pad_1d(b["history_y"], max_h, PAD_Y) for b in batch])

    deck = torch.stack([b["deck"] for b in batch])
    opp_deck = torch.stack([b["opp_deck"] for b in batch])

    y_gate = torch.stack([b["y_gate"] for b in batch])
    y_card = torch.stack([b["y_card"] for b in batch])

    target_x = torch.stack([b["target_x"] for b in batch])
    target_y = torch.stack([b["target_y"] for b in batch])
    has_xy = torch.stack([b["has_xy"] for b in batch])

    return {
        "history_cards": history_cards,
        "history_players": history_players,
        "history_x": history_x,
        "history_y": history_y,
        "deck": deck,
        "opp_deck": opp_deck,
        "y_gate": y_gate,
        "y_card": y_card,
        "target_x": target_x,
        "target_y": target_y,
        "has_xy": has_xy,
    }


@torch.no_grad()
def evaluate_gate(model: BCTransformer, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    pred_play = 0
    true_play = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model.forward_gate(
            history_cards=batch["history_cards"],
            history_players=batch["history_players"],
            deck=batch["deck"],
            opp_deck=batch["opp_deck"],
        )
        y = batch["y_gate"]
        loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * y.size(0)

        pred = logits.argmax(dim=-1)
        total += y.size(0)
        correct += (pred == y).sum().item()

        pred_play += (pred == 1).sum().item()
        true_play += (y == 1).sum().item()

    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
        "true_play_rate": true_play / max(1, total),
        "pred_play_rate": pred_play / max(1, total),
        "n": float(total),
    }


@torch.no_grad()
def evaluate_card(model: BCTransformer, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    top3_correct = 0

    true_noop = 0
    pred_noop = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(
            history_cards=batch["history_cards"],
            history_players=batch["history_players"],
            deck=batch["deck"],
            opp_deck=batch["opp_deck"],
        )
        y = batch["y_card"]
        loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * y.size(0)

        pred = logits.argmax(dim=-1)
        total += y.size(0)
        correct += (pred == y).sum().item()

        top3 = logits.topk(3, dim=-1).indices
        hit3 = (top3 == y.unsqueeze(1)).any(dim=1)
        top3_correct += hit3.sum().item()

        true_noop += (y == 8).sum().item()
        pred_noop += (pred == 8).sum().item()

    return {
        "loss": loss_sum / max(1, total),
        "acc": correct / max(1, total),
        "top3": top3_correct / max(1, total),
        "true_noop_rate": true_noop / max(1, total),
        "pred_noop_rate": pred_noop / max(1, total),
        "n": float(total),
    }


@torch.no_grad()
def evaluate_placement(model: BCTransformer, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate x/y only on samples where:
      - y_gate == PLAY
      - has_xy == 1
    """
    model.eval()
    total = 0
    correct_x = 0
    correct_y = 0
    correct_xy = 0
    loss_x_sum = 0.0
    loss_y_sum = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        gate_logits, card_logits, x_logits, y_logits = model.forward_policy(
            history_cards=batch["history_cards"],
            history_players=batch["history_players"],
            deck=batch["deck"],
            opp_deck=batch["opp_deck"],
            history_x=batch["history_x"],
            history_y=batch["history_y"],
        )

        mask = (batch["y_gate"] == 1) & (batch["has_xy"] == 1)
        if not mask.any():
            continue

        tx = batch["target_x"][mask]
        ty = batch["target_y"][mask]

        lx = x_logits[mask]
        ly = y_logits[mask]

        loss_x = F.cross_entropy(lx, tx)
        loss_y = F.cross_entropy(ly, ty)

        loss_x_sum += loss_x.item() * tx.size(0)
        loss_y_sum += loss_y.item() * ty.size(0)

        px = lx.argmax(dim=-1)
        py = ly.argmax(dim=-1)

        total += tx.size(0)
        correct_x += (px == tx).sum().item()
        correct_y += (py == ty).sum().item()
        correct_xy += ((px == tx) & (py == ty)).sum().item()

    return {
        "loss_x": loss_x_sum / max(1, total),
        "loss_y": loss_y_sum / max(1, total),
        "acc_x": correct_x / max(1, total),
        "acc_y": correct_y / max(1, total),
        "acc_xy": correct_xy / max(1, total),
        "n_play_xy": float(total),
    }


def main():
    parser = argparse.ArgumentParser()

    default_path = Path(os.getcwd()) / "bc_train.jsonl"
    parser.add_argument(
        "--train_jsonl",
        type=Path,
        default=default_path,
        help=f"Path to training bc_*.jsonl (default: {default_path})",
    )

    parser.add_argument(
        "--val_jsonl",
        type=Path,
        default=default_path,
        help=f"Path to validation bc_*.jsonl (default: {default_path})",
    )

    parser.add_argument(
        "--task",
        choices=["card", "gate", "joint"],
        default="card",
        help="card = train card head only, gate = train gate head only, joint = train gate+card+x+y",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--history_len", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=Path, default=Path("bc_model.pt"))

    # Streaming shuffle (bounded RAM)
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=5000,
        help="Bounded shuffle buffer for streaming dataset. 0 = no shuffle. Try 2000-20000.",
    )

    # For joint training only
    parser.add_argument("--lambda_gate", type=float, default=1.0, help="Weight on gate loss in joint training")
    parser.add_argument("--lambda_card", type=float, default=1.0, help="Weight on card loss in joint training")
    parser.add_argument("--lambda_x", type=float, default=1.0, help="Weight on x_tile loss in joint training")
    parser.add_argument("--lambda_y", type=float, default=1.0, help="Weight on y_tile loss in joint training")

    # Gate imbalance (PLAY is rare)
    parser.add_argument(
        "--gate_play_weight",
        type=float,
        default=20.0,
        help="Cross-entropy weight for PLAY class (WAIT weight is 1.0). Only used in joint/gate.",
    )

    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    train_files = list_or_build_jsonl(args.train_jsonl)
    val_files = list_or_build_jsonl(args.val_jsonl)

    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")

    # Build vocab from train only (streaming)
    token2id = build_vocab_from_jsonl_files(train_files)
    pad_id = token2id[PAD]
    unk_id = token2id[UNK]
    print(f"Vocab size: {len(token2id)}")

    # Data (streaming, constant memory)
    train_ds = BCJsonlIterableDataset(
        train_files,
        args.history_len,
        token2id,
        pad_id,
        unk_id,
        task=args.task,
        shuffle_buffer=args.shuffle_buffer,
    )
    val_ds = BCJsonlIterableDataset(
        val_files,
        args.history_len,
        token2id,
        pad_id,
        unk_id,
        task=args.task,
        shuffle_buffer=0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,  # IterableDataset cannot use shuffle=True
        collate_fn=lambda b: collate_dynamic(b, pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_dynamic(b, pad_id),
    )

    # Model
    model = BCTransformer(
        vocab_size=len(token2id),
        pad_id=pad_id,
        n_actions=9,  # 8 deck slots + NOOP
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if args.task == "gate":
                logits = model.forward_gate(
                    history_cards=batch["history_cards"],
                    history_players=batch["history_players"],
                    deck=batch["deck"],
                    opp_deck=batch["opp_deck"],
                )
                y = batch["y_gate"]
                gate_weights = torch.tensor([1.0, float(args.gate_play_weight)], device=device)
                loss = F.cross_entropy(logits, y, weight=gate_weights)

            elif args.task == "card":
                logits = model(
                    history_cards=batch["history_cards"],
                    history_players=batch["history_players"],
                    deck=batch["deck"],
                    opp_deck=batch["opp_deck"],
                )
                y = batch["y_card"]
                loss = F.cross_entropy(logits, y)

            elif args.task == "joint":
                gate_logits, card_logits, x_logits, y_logits = model.forward_policy(
                    history_cards=batch["history_cards"],
                    history_players=batch["history_players"],
                    deck=batch["deck"],
                    opp_deck=batch["opp_deck"],
                    history_x=batch["history_x"],
                    history_y=batch["history_y"],
                )

                y_gate = batch["y_gate"]
                y_card = batch["y_card"]

                gate_weights = torch.tensor([1.0, float(args.gate_play_weight)], device=device)
                loss_gate = F.cross_entropy(gate_logits, y_gate, weight=gate_weights)

                play_mask = (y_gate == 1)
                if play_mask.any():
                    loss_card = F.cross_entropy(card_logits[play_mask], y_card[play_mask])
                else:
                    loss_card = torch.tensor(0.0, device=device)

                place_mask = play_mask & (batch["has_xy"] == 1)
                if place_mask.any():
                    loss_x = F.cross_entropy(x_logits[place_mask], batch["target_x"][place_mask])
                    loss_y = F.cross_entropy(y_logits[place_mask], batch["target_y"][place_mask])
                else:
                    loss_x = torch.tensor(0.0, device=device)
                    loss_y = torch.tensor(0.0, device=device)

                loss = (
                    args.lambda_gate * loss_gate
                    + args.lambda_card * loss_card
                    + args.lambda_x * loss_x
                    + args.lambda_y * loss_y
                )

                logits = gate_logits
                y = y_gate

            else:
                raise ValueError(f"Unknown task: {args.task}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_count += y.size(0)

        train_loss = total_loss / max(1, total_count)
        train_acc = total_correct / max(1, total_count)

        if args.task == "gate":
            val = evaluate_gate(model, val_loader, device)
            print(
                f"[Epoch {epoch:02d}] "
                f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
                f"Val loss: {val['loss']:.4f}, acc: {val['acc']:.4f} | "
                f"true_play: {val['true_play_rate']:.3f}, pred_play: {val['pred_play_rate']:.3f} | "
                f"n={int(val['n'])}"
            )

        elif args.task == "card":
            val = evaluate_card(model, val_loader, device)
            print(
                f"[Epoch {epoch:02d}] "
                f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
                f"Val loss: {val['loss']:.4f}, acc: {val['acc']:.4f}, top3: {val['top3']:.4f} | "
                f"true_noop: {val['true_noop_rate']:.3f}, pred_noop: {val['pred_noop_rate']:.3f} | "
                f"n={int(val['n'])}"
            )

        else:  # joint
            val_g = evaluate_gate(model, val_loader, device)
            val_c = evaluate_card(model, val_loader, device)
            val_p = evaluate_placement(model, val_loader, device)
            print(
                f"[Epoch {epoch:02d}/{args.epochs}] "
                f"Train loss: {train_loss:.4f}, gate-acc: {train_acc:.4f} | "
                f"ValGate loss: {val_g['loss']:.4f}, acc: {val_g['acc']:.4f}, pred_play: {val_g['pred_play_rate']:.3f} | "
                f"ValCard loss: {val_c['loss']:.4f}, acc: {val_c['acc']:.4f}, top3: {val_c['top3']:.4f} | "
                f"ValXY acc_x: {val_p['acc_x']:.3f}, acc_y: {val_p['acc_y']:.3f}, acc_xy: {val_p['acc_xy']:.3f}, n_play_xy={int(val_p['n_play_xy'])}"
            )

    save_path = Path(args.save_path)

    # If save_path is a directory, save into it
    if save_path.exists() and save_path.is_dir():
        save_file = save_path / "model_state_dict.pt"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_file = save_path

    torch.save(model.state_dict(), str(save_file))
    print(f"Model saved to {save_file}")


if __name__ == "__main__":
    main()