import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ctx_max: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # Precompute causal mask once (True = allowed)
        causal = torch.tril(torch.ones(ctx_max, ctx_max, dtype=torch.bool))
        self.register_buffer("causal", causal, persistent=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, C)
        attn_mask: (B, T) bool, True = valid token (non-pad)
        """
        B, T, C = x.shape

        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, T, T)

        # --- causal mask ---
        causal = self.causal[:T, :T]  # (T, T) bool
        scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))

        # --- key padding mask (mask KEYS) ---
        if attn_mask is not None:
            key_ok = attn_mask[:, None, None, :]  # (B, 1, 1, T)
            scores = scores.masked_fill(~key_ok, float("-inf"))

        # --- SAFE softmax (prevents NaNs when a row is all -inf) ---
        w = F.softmax(scores, dim=-1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        w = self.drop(w)

        y = w @ v  # (B, H, T, dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ctx_max: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, ctx_max)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class BCTransformer(nn.Module):
    """
    BC transformer with:
      - Gate head: WAIT/PLAY
      - Card head: 8 deck slots + NOOP  => 9 logits
      - Placement heads (NEW): x_tile (18) and y_tile (32)

    Backwards compatible:
      - forward(...) returns card logits (B, 9) exactly like before
      - forward_gate(...) returns gate logits (B, 2) exactly like before

    New:
      - forward_policy(...) returns (gate, card, x, y) logits
      - supports optional history_x/history_y inside the encoder
    """

    def __init__(
        self,
        vocab_size: int,
        n_actions: int = 9,
        ctx_max: int = 256,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 6,
        dropout: float = 0.1,
        pad_id: int | None = None,  # not required but nice for clarity
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx_max = ctx_max
        # If caller didn't pass pad_id, default to 0 (matches your current training behavior)
        self.pad_id = 0 if pad_id is None else pad_id

        self.player_emb = nn.Embedding(2, d_model)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_max, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout, ctx_max) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

        # Heads
        self.head = nn.Linear(d_model, n_actions)     # Head B: deck-slot (0..7) + NOOP (8)
        self.gate_head = nn.Linear(d_model, 2)        # Head A: [WAIT, PLAY]

        # --- NEW: tile embeddings with explicit PAD ids ---
        # Valid ranges:
        #   x_tile: 0..17, pad_x = 18
        #   y_tile: 0..31, pad_y = 32
        self.pad_x = 18
        self.pad_y = 32
        self.x_emb = nn.Embedding(19, d_model, padding_idx=self.pad_x)  # 18 + PAD
        self.y_emb = nn.Embedding(33, d_model, padding_idx=self.pad_y)  # 32 + PAD

        # --- NEW: placement heads ---
        self.x_head = nn.Linear(d_model, 18)
        self.y_head = nn.Linear(d_model, 32)

    def _encode_last_h(
        self,
        history_cards: torch.Tensor,
        history_players: torch.Tensor,
        deck: torch.Tensor,
        opp_deck: torch.Tensor,
        history_x: Optional[torch.Tensor] = None,
        history_y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns last_h: (B, d_model)

        Inputs:
          history_cards:   (B, H) token ids, padded with self.pad_id
          history_players: (B, H) 0=team, 1=opp
          deck:            (B, 8)
          opp_deck:        (B, 8)
          history_x:       (B, H) x_tile in [0..17] or pad_x (=18). Optional.
          history_y:       (B, H) y_tile in [0..31] or pad_y (=32). Optional.
        """
        B, H = history_cards.shape

        # True for valid (non-pad) history tokens
        attn_mask = (history_cards != self.pad_id)  # (B, H)

        # Prevent all-pad rows from creating all -inf attention
        all_pad = ~attn_mask.any(dim=1)  # (B,)
        if all_pad.any():
            attn_mask = attn_mask.clone()
            attn_mask[all_pad, 0] = True

        # Defaults for x/y if not provided yet (backwards compatibility)
        if history_x is None:
            history_x = torch.full((B, H), self.pad_x, dtype=torch.long, device=history_cards.device)
        if history_y is None:
            history_y = torch.full((B, H), self.pad_y, dtype=torch.long, device=history_cards.device)

        # --- embed history cards ---
        h_card = self.tok_emb(history_cards)  # (B, H, d_model)

        # --- embed player id (0/1) ---
        h_pl = self.player_emb(history_players.clamp(0, 1))  # (B, H, d_model)

        # --- NEW: embed placement tiles ---
        h_x = self.x_emb(history_x.clamp(0, self.pad_x))  # (B, H, d_model)
        h_y = self.y_emb(history_y.clamp(0, self.pad_y))  # (B, H, d_model)

        # --- embed decks and broadcast to history length ---
        d_me = self.tok_emb(deck).mean(dim=1, keepdim=True)       # (B, 1, d_model)
        d_opp = self.tok_emb(opp_deck).mean(dim=1, keepdim=True)  # (B, 1, d_model)

        x = h_card + h_pl + h_x + h_y + d_me + d_opp  # (B, H, d_model)

        # positions
        if H > self.ctx_max:
            raise ValueError(f"history_len {H} > ctx_max {self.ctx_max}")
        pos = torch.arange(H, device=x.device).unsqueeze(0).expand(B, H)
        x = x + self.pos_emb(pos)

        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.ln_f(x)

        # readout: last valid position in history (or 0 if all pad)
        last_idx = (attn_mask.long().sum(dim=1) - 1).clamp(min=0)
        last_h = x[torch.arange(B, device=x.device), last_idx]  # (B, d_model)
        return last_h

    # ---- Backwards compatible forward: returns ONLY card logits like before ----
    def forward(
        self,
        history_cards: torch.Tensor,
        history_players: torch.Tensor,
        deck: torch.Tensor,
        opp_deck: torch.Tensor,
    ) -> torch.Tensor:
        """
        Output:
          logits: (B, 9)  -> 0..7 deck slots, 8 NOOP
        """
        last_h = self._encode_last_h(history_cards, history_players, deck, opp_deck)
        logits = self.head(last_h)  # (B, 9)
        return logits

    # ---- Backwards compatible gate forward: returns ONLY gate logits like before ----
    def forward_gate(
        self,
        history_cards: torch.Tensor,
        history_players: torch.Tensor,
        deck: torch.Tensor,
        opp_deck: torch.Tensor,
    ) -> torch.Tensor:
        """
        Head A: play gate
        Output logits: (B, 2) = [WAIT, PLAY]
        """
        last_h = self._encode_last_h(history_cards, history_players, deck, opp_deck)
        logits_gate = self.gate_head(last_h)  # (B, 2)
        return logits_gate

    # ---- NEW: policy forward that includes placement ----
    def forward_policy(
        self,
        history_cards: torch.Tensor,
        history_players: torch.Tensor,
        deck: torch.Tensor,
        opp_deck: torch.Tensor,
        history_x: torch.Tensor,
        history_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          gate_logits: (B, 2)   [WAIT, PLAY]
          card_logits: (B, 9)   0..7 deck slots, 8 NOOP
          x_logits:    (B, 18)  x_tile
          y_logits:    (B, 32)  y_tile
        """
        last_h = self._encode_last_h(
            history_cards=history_cards,
            history_players=history_players,
            deck=deck,
            opp_deck=opp_deck,
            history_x=history_x,
            history_y=history_y,
        )
        gate_logits = self.gate_head(last_h)
        card_logits = self.head(last_h)
        x_logits = self.x_head(last_h)
        y_logits = self.y_head(last_h)
        return gate_logits, card_logits, x_logits, y_logits
