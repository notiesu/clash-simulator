# train.py
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import BCTransformer


PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
NOOP = "NOOP"


def list_jsonl_files(p: Path) -> List[Path]:
    """
    Accepts a .jsonl file or a directory.
    Directory mode loads ONLY bc_*.jsonl (your desired format).
    """
    p = Path(p)
    if p.is_file():
        if p.name.startswith("bc_") and p.suffix == ".jsonl":
            return [p]
        raise ValueError(f"File does not match bc_*.jsonl pattern: {p}")

    if p.is_dir():
        files = sorted(p.glob("bc_*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No bc_*.jsonl files found in: {p}")
        return files

    raise FileNotFoundError(f"{p} is not a file or directory")


def load_rows(files: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def build_vocab_from_train(train_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    vocab = {PAD, UNK, BOS, EOS, NOOP}

    for r in train_rows:
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

        # label (sometimes useful for vocab)
        lbl = r.get("label")
        if isinstance(lbl, str) and lbl:
            vocab.add(lbl)

    token2id = {tok: i for i, tok in enumerate(sorted(vocab))}
    return token2id


class BCRowsDataset(torch.utils.data.Dataset):
    """
    Works for:
      - gate task: y_gate in {0,1}
      - card task: y_card in {0..8} where 8 is NOOP fallback
    """
    def __init__(self, rows: List[Dict[str, Any]], history_len: int,
                 token2id: Dict[str, int], pad_id: int, unk_id: int):
        self.rows = rows
        self.history_len = history_len
        self.token2id = token2id
        self.pad_id = pad_id
        self.unk_id = unk_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.rows[idx]

        # ---------------------------
        # HISTORY (fixed length)
        # ---------------------------
        history = sample.get("history", [])
        cards: List[int] = []
        players: List[int] = []

        if isinstance(history, list):
            for h in history:
                if not isinstance(h, dict):
                    continue
                card = h.get("card", UNK)
                p = h.get("p", "team")
                cards.append(self.token2id.get(card, self.unk_id))
                players.append(0 if p == "team" else 1)

        cards = cards[-self.history_len:]
        players = players[-self.history_len:]
        m = min(len(cards), len(players))
        cards = cards[-m:]
        players = players[-m:]

        pad_n = self.history_len - len(cards)
        if pad_n > 0:
            cards = [self.pad_id] * pad_n + cards
            players = [0] * pad_n + players

        history_cards = torch.tensor(cards, dtype=torch.long)
        history_players = torch.tensor(players, dtype=torch.long)

        # ---------------------------
        # DECKS (preserve order)
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
        # LABELS
        # ---------------------------
        lbl = sample.get("label", NOOP)
        is_play = isinstance(lbl, str) and lbl and (lbl != NOOP)

        # Head A: gate label
        y_gate = torch.tensor(1 if is_play else 0, dtype=torch.long)  # 0=WAIT, 1=PLAY

        # Head B: slot label (Option A)
        action_list = deck + [NOOP]  # 9 actions (8 deck slots + NOOP)
        label = lbl if isinstance(lbl, str) and lbl else NOOP
        if label not in action_list:
            label = NOOP
        y_card = torch.tensor(action_list.index(label), dtype=torch.long)  # 0..8

        return {
            "history_cards": history_cards,
            "history_players": history_players,
            "deck": deck_ids,
            "opp_deck": opp_deck_ids,
            "y_gate": y_gate,
            "y_card": y_card,
        }


def collate_dynamic(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    # Dynamic pad history in case some sample slipped through with shorter tensors
    def pad_1d(t: torch.Tensor, L: int, pad_value: int) -> torch.Tensor:
        if t.numel() == L:
            return t
        if t.numel() > L:
            return t[-L:]
        out = torch.full((L,), pad_value, dtype=t.dtype)
        out[-t.numel():] = t
        return out

    max_h = max(b["history_cards"].numel() for b in batch)

    history_cards = torch.stack([pad_1d(b["history_cards"], max_h, pad_id) for b in batch])
    history_players = torch.stack([pad_1d(b["history_players"], max_h, 0) for b in batch])

    deck = torch.stack([b["deck"] for b in batch])
    opp_deck = torch.stack([b["opp_deck"] for b in batch])

    y_gate = torch.stack([b["y_gate"] for b in batch])
    y_card = torch.stack([b["y_card"] for b in batch])

    return {
        "history_cards": history_cards,
        "history_players": history_players,
        "deck": deck,
        "opp_deck": opp_deck,
        "y_gate": y_gate,
        "y_card": y_card,
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

    # track how often NOOP shows up in labels/preds (should be 0 if you filtered to plays)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=Path, required=True,
                        help="Path to a bc_*.jsonl file OR a directory containing bc_*.jsonl files")
    parser.add_argument("--val_jsonl", type=Path, required=True,
                        help="Path to a bc_*.jsonl file OR a directory containing bc_*.jsonl files")

    parser.add_argument("--task", choices=["card", "gate", "joint"], default="card",
                        help="card = train card head only, gate = train gate head only, joint = train both")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--history_len", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=Path, default=Path("bc_model.pt"))

    # For joint training only
    parser.add_argument("--lambda_gate", type=float, default=1.0,
                        help="Weight on gate loss in joint training")
    parser.add_argument("--lambda_card", type=float, default=1.0,
                        help="Weight on card loss in joint training")

    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    train_files = list_jsonl_files(args.train_jsonl)
    val_files = list_jsonl_files(args.val_jsonl)

    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")

    train_rows = load_rows(train_files)
    val_rows = load_rows(val_files)

    # Optional: for card-only training, you may want to filter to true plays-in-deck
    if args.task == "card":
        def is_good_play_row(r: Dict[str, Any]) -> bool:
            lbl = r.get("label", None)
            if not (isinstance(lbl, str) and lbl and lbl != NOOP):
                return False
            deck = r.get("deck", [])
            return isinstance(deck, list) and (lbl in deck)

        train_rows = [r for r in train_rows if is_good_play_row(r)]
        val_rows = [r for r in val_rows if is_good_play_row(r)]
        print(f"Train PLAY samples (lines): {len(train_rows)}")
        print(f"Val PLAY samples (lines):   {len(val_rows)}")
    else:
        print(f"Train samples (lines): {len(train_rows)}")
        print(f"Val samples (lines):   {len(val_rows)}")

    # Build vocab from train only
    token2id = build_vocab_from_train(train_rows)
    pad_id = token2id[PAD]
    unk_id = token2id[UNK]

    # Data
    train_ds = BCRowsDataset(train_rows, args.history_len, token2id, pad_id, unk_id)
    val_ds = BCRowsDataset(val_rows, args.history_len, token2id, pad_id, unk_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_dynamic(b, pad_id)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_dynamic(b, pad_id)
    )

    # Model
    model = BCTransformer(
        vocab_size=len(token2id),
        pad_id=pad_id,
        n_actions=9,  # for card head output logits (0..7 + NOOP)
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
                loss = F.cross_entropy(logits, y)

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
                # -------- Gate head --------
                logits_gate = model.forward_gate(
                    history_cards=batch["history_cards"],
                    history_players=batch["history_players"],
                    deck=batch["deck"],
                    opp_deck=batch["opp_deck"],
                )
                y_gate = batch["y_gate"]  # 0 = WAIT, 1 = PLAY

                # IMPORTANT: class-weighted gate loss (PLAY is rare)
                gate_weights = torch.tensor([1.0, 20.0], device=device)  # [WAIT, PLAY]
                loss_gate = F.cross_entropy(logits_gate, y_gate, weight=gate_weights)

                # -------- Card head (ONLY on PLAY samples) --------
                is_play = (y_gate == 1)

                if is_play.any():
                    logits_card = model(
                        history_cards=batch["history_cards"][is_play],
                        history_players=batch["history_players"][is_play],
                        deck=batch["deck"][is_play],
                        opp_deck=batch["opp_deck"][is_play],
                    )
                    y_card = batch["y_card"][is_play]
                    loss_card = F.cross_entropy(logits_card, y_card)
                else:
                    loss_card = torch.tensor(0.0, device=device)

                # -------- Combined loss --------
                loss = args.lambda_gate * loss_gate + args.lambda_card * loss_card

                # For logging accuracy, report gate performance
                logits = logits_gate
                y = y_gate

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

        else:  # joint: print both evals
            val_g = evaluate_gate(model, val_loader, device)
            val_c = evaluate_card(model, val_loader, device)
            print(
                f"[Epoch {epoch:02d}] "
                f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
                f"ValGate loss: {val_g['loss']:.4f}, acc: {val_g['acc']:.4f}, pred_play: {val_g['pred_play_rate']:.3f} | "
                f"ValCard loss: {val_c['loss']:.4f}, acc: {val_c['acc']:.4f}, top3: {val_c['top3']:.4f} | "
                f"n={int(val_g['n'])}"
            )

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
