import math
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
        self.pad_id = pad_id

        self.player_emb = nn.Embedding(2, d_model)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_max, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout, ctx_max) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, n_actions)
        self.gate_head = nn.Linear(d_model, 2)  # Head A: [WAIT, PLAY]

    def forward(
        self,
        history_cards: torch.Tensor,
        history_players: torch.Tensor,
        deck: torch.Tensor,
        opp_deck: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
          history_cards:   (B, H) LongTensor token ids (padded with pad_id)
          history_players: (B, H) LongTensor, 0=team, 1=opp
          deck:            (B, 8) LongTensor token ids
          opp_deck:        (B, 8) LongTensor token ids

        Output:
          logits: (B, 9)  -> 0..7 deck slots, 8 NOOP
        """
        B, H = history_cards.shape

        # True for valid (non-pad) history tokens
        attn_mask = (history_cards != self.pad_id)  # (B, H)

        # --- IMPORTANT: prevent all-pad rows from creating all -inf attention ---
        all_pad = ~attn_mask.any(dim=1)  # (B,)
        if all_pad.any():
            attn_mask = attn_mask.clone()
            attn_mask[all_pad, 0] = True

        # --- embed history cards ---
        h_card = self.tok_emb(history_cards)  # (B, H, d_model)

        # --- embed player id (0/1) ---
        h_pl = self.player_emb(history_players.clamp(0, 1))  # (B, H, d_model)

        # --- embed decks and broadcast to history length ---
        d_me = self.tok_emb(deck).mean(dim=1, keepdim=True)       # (B, 1, d_model)
        d_opp = self.tok_emb(opp_deck).mean(dim=1, keepdim=True)  # (B, 1, d_model)

        x = h_card + h_pl + d_me + d_opp  # (B, H, d_model)

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

        logits = self.head(last_h)  # (B, 9)
        return logits
    
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
        B, H = history_cards.shape

        attn_mask = (history_cards != self.pad_id)
        all_pad = ~attn_mask.any(dim=1)
        if all_pad.any():
            attn_mask = attn_mask.clone()
            attn_mask[all_pad, 0] = True

        h_card = self.tok_emb(history_cards)
        h_pl = self.player_emb(history_players.clamp(0, 1))

        d_me = self.tok_emb(deck).mean(dim=1, keepdim=True)
        d_opp = self.tok_emb(opp_deck).mean(dim=1, keepdim=True)

        x = h_card + h_pl + d_me + d_opp

        if H > self.ctx_max:
            raise ValueError(f"history_len {H} > ctx_max {self.ctx_max}")
        pos = torch.arange(H, device=x.device).unsqueeze(0).expand(B, H)
        x = x + self.pos_emb(pos)

        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.ln_f(x)

        last_idx = (attn_mask.long().sum(dim=1) - 1).clamp(min=0)
        last_h = x[torch.arange(B, device=x.device), last_idx]  # (B, d_model)

        logits_gate = self.gate_head(last_h)  # (B, 2)
        return logits_gate
