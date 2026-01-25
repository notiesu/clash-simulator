import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

# =========================
# Special tokens (must match tokenize.py / train.py)
# =========================
PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
NOOP = "NOOP"


class BehaviorCloningDataset(Dataset):
    """
    Behavior Cloning Dataset (Option A: Slot-Based Actions)

    Action space per sample:
        0..7 -> play deck slot i
        8    -> NOOP

    IMPORTANT:
    - Deck order is PRESERVED exactly as stored in the data.
    - Slot indices are stable for the entire match.
    """

    def __init__(self, jsonl_path: Path, history_len: int = 20):
        self.jsonl_path = Path(jsonl_path)
        self.history_len = history_len

        # -------------------------
        # Load JSONL
        # -------------------------
        self.rows: List[Dict[str, Any]] = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

        # -------------------------
        # Build token vocabulary (cards only)
        # -------------------------
        vocab = {PAD, UNK, BOS, EOS, NOOP}

        for r in self.rows:
            # history cards
            for h in r.get("history", []):
                c = h.get("card")
                if c:
                    vocab.add(c)

            # own deck (slot order preserved)
            for c in r.get("deck", []):
                if c:
                    vocab.add(c)

            # opponent deck
            for c in r.get("opp_deck", []):
                if c:
                    vocab.add(c)

            # labels
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

        # ======================================================
        # 1) HISTORY (cards + player identity)
        # ======================================================
        history = sample.get("history", [])

        cards: List[int] = []
        players: List[int] = []

        for h in history[-self.history_len :]:
            card = h.get("card", UNK)
            player = h.get("p", "team")  # "team" or "opp"

            cards.append(self.token2id.get(card, self.unk_id))
            players.append(0 if player == "team" else 1)

        # left-pad history
        pad_n = self.history_len - len(cards)
        if pad_n > 0:
            cards = [self.pad_id] * pad_n + cards
            players = [0] * pad_n

        history_cards = torch.tensor(cards, dtype=torch.long)
        history_players = torch.tensor(players, dtype=torch.long)

        # ======================================================
        # 2) DECKS (slot order PRESERVED)
        # ======================================================
        deck = sample.get("deck", [])
        deck = [c for c in deck if c]

        if len(deck) != 8:
            deck = (deck + [UNK] * 8)[:8]

        opp_deck = sample.get("opp_deck", [])
        opp_deck = [c for c in opp_deck if c]

        if len(opp_deck) != 8:
            opp_deck = (opp_deck + [UNK] * 8)[:8]

        deck_ids = torch.tensor(
            [self.token2id.get(c, self.unk_id) for c in deck],
            dtype=torch.long,
        )

        opp_deck_ids = torch.tensor(
            [self.token2id.get(c, self.unk_id) for c in opp_deck],
            dtype=torch.long,
        )

        # ======================================================
        # 3) ACTION LABEL (slot-based: 0..8)
        # ======================================================
        action_list = deck + [NOOP]  # slot 0..7 + NOOP

        label = sample.get("label", NOOP)
        if label not in action_list:
            label = NOOP

        y = action_list.index(label)  # ALWAYS in [0..8]

        # ======================================================
        # 4) RETURN
        # ======================================================
        return {
            "history_cards": history_cards,
            "history_players": history_players,
            "deck": deck_ids,
            "opp_deck": opp_deck_ids,
            "y": torch.tensor(y, dtype=torch.long),
        }


def collate_pad(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "history_cards": torch.stack([b["history_cards"] for b in batch]),
        "history_players": torch.stack([b["history_players"] for b in batch]),
        "deck": torch.stack([b["deck"] for b in batch]),
        "opp_deck": torch.stack([b["opp_deck"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }
