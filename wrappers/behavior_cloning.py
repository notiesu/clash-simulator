from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import json
import numpy as np
import torch

from src.clasher.model import InferenceModel
from bc_transformer.train.model import BCTransformer


# ----------------------------
# Card name mapping
# bc_model token -> gym env card name
# ----------------------------
MODEL_TO_ENV = {
                    "cannon"     : "Cannon",
                    "fireball"   : "Fireball",
                    "hog-rider"  : "HogRider",
                    "ice-golem"  : "IceGolemite",
                    "ice-spirit" : "IceSpirits",
                    "musketeer"  : "Musketeer",
                    "skeletons"  : "Skeletons",
                    "the-log"    : "Log",
                }
ENV_TO_MODEL = {v: k for k, v in MODEL_TO_ENV.items()}


@dataclass
class BCArgs:
    """
    Runtime settings for BC inference.

    NOTE:
      If your .pt file is ONLY a raw state_dict, we still need vocab_size/pad_id to rebuild the model.
      This class lets inference.py pass those in, BUT if inference.py doesn't pass them, load_model()
      will try to auto-discover:
        - token2id embedded in checkpoint (preferred), or
        - a sidecar JSON next to the weights file (token2id.json / *.token2id.json), or
        - env.token2id / env.vocab_size (last resort).
    """
    token2id_path: Optional[str] = None   # path to token2id.json
    pad_id: Optional[int] = None          # pad token id used during training
    device: str = "cpu"                   # "cpu" or "cuda"
    history_len: int = 20                 # number of past steps to feed
    use_gate: bool = False                # if your model has forward_gate()


class BCInferenceModel(InferenceModel):
    """
    Env-facing Behavioral Cloning inference wrapper.

    API matches inference.py expectations:
      model = BCInferenceModel()
      model.load_model(path)
      obs_p0 = model.preprocess_observation(obs)
      a = model.predict(obs_p0)
      a = model.postprocess_action(a)

    Returns a single discrete action:
      0..7 = choose deck slot
      8    = NOOP / wait
    """

    def __init__(self, env=None, bc_args: Optional[BCArgs] = None, printLogs: bool = False):
        super().__init__()

        self.env = env
        self.printLogs = printLogs
        self.bc_args = bc_args or BCArgs()
        self.model: Optional[BCTransformer] = None
        self.token2id: Optional[Dict[str, int]] = None
        self.pad_id: Optional[int] = self.bc_args.pad_id

        # resolve device now; will be used in load_model too
        dev = self.bc_args.device
        self.device = torch.device(dev if (dev.startswith("cuda") and torch.cuda.is_available()) else "cpu")

        # ----------------------------
        # AUTO-LOAD at init (no inference.py changes)
        # ----------------------------
        # We can always find the weights path from the token2id_path directory.
        # inference.py passes token2id_path but never calls load_model() :contentReference[oaicite:2]{index=2}
        t2i_path = getattr(self.bc_args, "token2id_path", None)
        if t2i_path:
            t2i_path = Path(t2i_path)

            # Candidate weight filenames (pick the first one that exists)
            weight_candidates = [
                t2i_path.parent / "model_state_dict.pt",
                t2i_path.parent / "bc_model.pt",
                t2i_path.parent / "model.pt",
                t2i_path.parent / "checkpoint.pt",
            ]

            model_path = None
            for c in weight_candidates:
                if c.exists():
                    model_path = c
                    break

            if model_path is None:
                raise FileNotFoundError(
                    f"BCInferenceModel init: couldn't find weights next to token2id.json.\n"
                    f"token2id_path={t2i_path}\n"
                    f"Tried: {[str(x) for x in weight_candidates]}"
                )

            # This will load token2id + build model + load weights
            self.load_model(model_path)
        else:
            # If you didn't pass token2id_path, we can't auto-load.
            # (Inference.py currently doesn't call load_model for BC.)
            if self.printLogs:
                print("[BCInferenceModel] WARNING: bc_args.token2id_path is None, so model was NOT auto-loaded.")

    def _env_card_to_model_token(self, env_card: str) -> str:
        # gym env name -> bc token
        return ENV_TO_MODEL.get(env_card, env_card)

    def _model_token_to_env_card(self, model_token: str) -> str:
        # bc token -> gym env name
        return MODEL_TO_ENV.get(model_token, model_token)

    def _encode_cards_to_ids(self, cards) -> torch.Tensor:
        """
        cards: list[str] (env names or bc tokens) OR list[int] OR ndarray/tensor ints
        returns: 1D torch.long token IDs for the BC model
        """
        # numeric already
        if isinstance(cards, torch.Tensor):
            if cards.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
                return cards.view(-1).long()
        if isinstance(cards, np.ndarray) and np.issubdtype(cards.dtype, np.integer):
            return torch.from_numpy(cards).view(-1).long()
        if isinstance(cards, list) and (len(cards) == 0 or isinstance(cards[0], (int, np.integer))):
            return torch.tensor(cards, dtype=torch.long).view(-1)

        # strings -> token ids
        if self.token2id is None:
            raise ValueError("token2id not loaded (need token2id.json or embedded in checkpoint).")

        unk = self.token2id.get("<UNK>", int(self.pad_id))
        ids = []
        for c in cards:
            if c is None:
                ids.append(int(self.pad_id))
                continue
            if isinstance(c, str):
                model_tok = self._env_card_to_model_token(c)
                ids.append(self.token2id.get(model_tok, unk))
            else:
                ids.append(int(c))
        return torch.tensor(ids, dtype=torch.long).view(-1)

    def _choose_tile_index(self, model_token: str) -> int:
        """
        Heuristic tile chooser (player-0 coordinate system).
        Player-1 y flip happens inside env.decode_and_deploy, so we DO NOT flip here.
        """
        u = getattr(self.env, "unwrapped", self.env)
        tiles_x = int(getattr(u, "tiles_x", 1))
        tiles_y = int(getattr(u, "tiles_y", 1))
        actions_per_tile = int(getattr(u, "actions_per_tile", tiles_x * tiles_y))

        # center x
        x = tiles_x // 2

        # choose y by card type
        tok = (model_token or "").lower()
        if tok in ("fireball", "the-log"):
            # spells: aim mid-enemy side
            y = int(0.70 * (tiles_y - 1))
        elif tok in ("cannon",):
            # building: defensive placement near our side
            y = int(0.25 * (tiles_y - 1))
        else:
            # troops: near bridge-ish / mid
            y = int(0.45 * (tiles_y - 1))

        # clamp
        x = max(0, min(tiles_x - 1, x))
        y = max(0, min(tiles_y - 1, y))

        tile_index = y * tiles_x + x
        tile_index = max(0, min(actions_per_tile - 1, tile_index))
        return tile_index

    def _infer_player_id(self) -> int:
        """
        No inference.py changes:
        - If this model is being used as env.opponent_policy, we are player 1.
        - Otherwise, assume player 0.
        """
        u = getattr(self.env, "unwrapped", self.env)
        if hasattr(u, "opponent_policy") and getattr(u, "opponent_policy") is self:
            return 1
        return 0

    # --------------------------
    # Interface required by inference.py
    # --------------------------
    def wrap_env(self, env):
        self.env = env
        return env

    def reset(self):
        return self.env.reset()

    def load_model(self, model_path: Union[str, Path]):

        model_path = Path(model_path)

        # -----------------------------
        # 1) Load checkpoint / state_dict
        # -----------------------------
        ckpt = torch.load(str(model_path), map_location=self.device)

        # accept either raw state_dict or {"state_dict": ...}
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        # -----------------------------
        # 2) Load token2id (path -> embedded -> sidecar -> env)
        # -----------------------------
        token2id = None

        # (a) explicit vocab path from bc_args (CLI: --p0_vocab_json)
        t2i_path = getattr(self.bc_args, "token2id_path", None)
        if t2i_path:
            p = Path(t2i_path)
            if not p.exists():
                raise FileNotFoundError(f"token2id_path does not exist: {p}")
            with open(p, "r") as f:
                token2id = json.load(f)

        # (b) embedded in ckpt (if training saved it)
        if token2id is None and isinstance(ckpt, dict):
            for k in ("token2id", "token_to_id", "vocab", "vocab_dict"):
                if k in ckpt and isinstance(ckpt[k], dict) and len(ckpt[k]) > 0:
                    token2id = ckpt[k]
                    break

        # (c) sidecar next to weights
        if token2id is None:
            sidecar = model_path.parent / "token2id.json"
            if sidecar.exists():
                with open(sidecar, "r") as f:
                    token2id = json.load(f)

        # (d) last resort: env provides it
        if token2id is None and self.env is not None:
            u = getattr(self.env, "unwrapped", self.env)
            maybe = getattr(u, "token2id", None)
            if isinstance(maybe, dict) and len(maybe) > 0:
                token2id = maybe

        if token2id is None:
            raise ValueError(
                "token2id not loaded. Provide bc_args.token2id_path (from --p0_vocab_json), "
                "or place token2id.json next to the model weights, or embed token2id in checkpoint."
            )

        # normalize values to int
        self.token2id = {str(k): int(v) for k, v in token2id.items()}

        # -----------------------------
        # 3) Determine vocab size expected by checkpoint
        # -----------------------------
        emb_key = None
        if "tok_emb.weight" in state:
            emb_key = "tok_emb.weight"
        elif "model.tok_emb.weight" in state:
            emb_key = "model.tok_emb.weight"

        if emb_key is None:
            # fallback: trust token2id size (not ideal but avoids crash)
            ckpt_vocab_size = len(self.token2id)
        else:
            ckpt_vocab_size = int(state[emb_key].shape[0])

        vocab_size = ckpt_vocab_size

        # -----------------------------
        # 4) Pad token2id to match checkpoint vocab size (if smaller)
        # -----------------------------
        if len(self.token2id) < vocab_size:
            used_ids = set(self.token2id.values())
            # Fill any missing ids in 0..vocab_size-1 with placeholder tokens
            for tid in range(vocab_size):
                if tid in used_ids:
                    continue
                name = f"<EXTRA_{tid}>"
                while name in self.token2id:
                    name += "_"
                self.token2id[name] = tid
                used_ids.add(tid)

        # -----------------------------
        # 5) Resolve pad_id
        # -----------------------------
        if self.pad_id is None:
            self.pad_id = int(self.token2id.get("<PAD>", 0))

        # -----------------------------
        # 6) Build model (MUST match training hyperparams)
        # -----------------------------
        m = BCTransformer(
            vocab_size=vocab_size,   # <-- IMPORTANT: match checkpoint embedding
            n_actions=9,
            ctx_max=256,
            d_model=192,
            n_heads=6,
            n_layers=6,
            dropout=0.1,
            pad_id=int(self.pad_id),
        ).to(self.device)

        # -----------------------------
        # 7) Load weights
        # -----------------------------
        m.load_state_dict(state, strict=True)
        m.eval()
        self.model = m

        if self.printLogs:
            src = str(t2i_path) if t2i_path else "sidecar/env/embedded"
            print(f"[BCInferenceModel] Loaded weights: {model_path}")
            print(f"[BCInferenceModel] token2id: {src} | ckpt_vocab={vocab_size} | pad_id={self.pad_id} | device={self.device}")

    def update_history_from_info(self, info):
        if not hasattr(self, "_hist_cards"):
            from collections import deque
            self._hist_cards = deque(maxlen=self.bc_args.history_len)
            self._hist_players = deque(maxlen=self.bc_args.history_len)

        last = info.get("last_action", None)
        if last is None:
            return

        for player_key, player_id in [("player_0", 0), ("player_1", 1)]:
            action = last.get(player_key, {})
            card_name = action.get("card_name", None)
            success = action.get("success", False)

            if success and card_name and card_name != "None":
                model_card = ENV_TO_MODEL.get(card_name, None)
                if model_card is None:
                    print("UNKNOWN ENV CARD:", repr(card_name), "not in ENV_TO_MODEL")
                    return  # or continue
                self._hist_cards.append(model_card)
                self._hist_players.append(player_id)

    def preprocess_observation(self, observation: Any) -> Dict[str, torch.Tensor]:
        """
        BC expects tokens; env currently returns pixels (H,W,3).

        This function will build a BC token observation from env state:
          - deck / opp_deck from current hands
          - history_cards/history_players from internal buffers if available;
            otherwise pads with <PAD>

        Returns batch dict on device:
          history_cards:   (1,H)
          history_players: (1,H)
          deck:            (1,hand_size)
          opp_deck:        (1,hand_size)
        """
        u = getattr(self.env, "unwrapped", self.env)

        # Try to fetch structured obs if env provides it
        obs_dict = None
        for name in ("get_bc_obs", "get_token_obs", "get_token_observation", "get_obs_dict"):
            if hasattr(u, name) and callable(getattr(u, name)):
                try:
                    out = getattr(u, name)()
                    if isinstance(out, dict):
                        obs_dict = out
                        break
                except Exception:
                    pass

        # If still None, synthesize from env battle state (hands)
        if obs_dict is None:
            if not hasattr(u, "battle"):
                raise TypeError(
                    "Env obs is pixels, and env doesn't expose battle state or get_bc_obs().\n"
                    "Add env.get_bc_obs() returning keys: history_cards/history_players/deck/opp_deck."
                )

            p0_hand = list(u.battle.players[0].hand)
            p1_hand = list(u.battle.players[1].hand)

            # Optional: if env stores history buffers, use them; else empty
            hist_cards = getattr(self, "_hist_cards", [])
            hist_players = getattr(self, "_hist_players", [])

            print("RAW env history_cards:", hist_cards)

            obs_dict = {
                "history_cards": hist_cards if hist_cards is not None else [],
                "history_players": hist_players if hist_players is not None else [],
                "deck": p0_hand,
                "opp_deck": p1_hand,
            }

        # normalize keys
        h_cards = obs_dict.get("history_cards", obs_dict.get("history", []))
        h_players = obs_dict.get("history_players", obs_dict.get("players", []))
        deck = obs_dict.get("deck")
        opp_deck = obs_dict.get("opp_deck")

        if deck is None or opp_deck is None:
            raise KeyError(f"BC obs dict must include 'deck' and 'opp_deck'. Got keys={list(obs_dict.keys())}")

        def to_long_1d(x) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                t = x
            elif isinstance(x, np.ndarray):
                t = torch.from_numpy(x)
            else:
                t = torch.tensor(x)
            return t.view(-1).long()

        # map env card names -> model token ids
        history_cards = self._encode_cards_to_ids(h_cards if h_cards is not None else [])
        deck_ids = self._encode_cards_to_ids(deck)
        opp_deck_ids = self._encode_cards_to_ids(opp_deck)

        # players history (0/1); if missing, default zeros matching history_cards length
        if h_players is None or len(h_players) == 0:
            history_players = torch.zeros_like(history_cards)
        else:
            history_players = to_long_1d(h_players)

        # pad/trim history to fixed length
        H = int(self.bc_args.history_len)
        if history_cards.numel() > H:
            history_cards = history_cards[-H:]
            history_players = history_players[-H:]
        elif history_cards.numel() < H:
            pad_n = H - history_cards.numel()
            history_cards = torch.cat(
                [torch.full((pad_n,), int(self.pad_id), dtype=torch.long), history_cards], dim=0
            )
            history_players = torch.cat(
                [torch.zeros((pad_n,), dtype=torch.long), history_players], dim=0
            )

        batch = {
            "history_cards": history_cards.unsqueeze(0).to(self.device),
            "history_players": history_players.unsqueeze(0).to(self.device),
            "deck": deck_ids.view(1, -1).to(self.device),
            "opp_deck": opp_deck_ids.view(1, -1).to(self.device),
        }
        return batch

    def postprocess_action(self, model_output: Any) -> int:
        """
        Convert model output -> env encoded action integer.

        NEW expected model output:
          -1        = NOOP / WAIT
          0..3      = hand slot index (indexes CURRENT HAND)

        Env expects:
          action = card_idx * actions_per_tile + tile_index
        where card_idx indexes CURRENT HAND: battle.players[player_id].hand
        """
        a = int(model_output[0] if isinstance(model_output, tuple) else model_output)

        u = getattr(self.env, "unwrapped", self.env)
        actions_per_tile = int(
            getattr(u, "actions_per_tile", getattr(u, "tiles_x", 1) * getattr(u, "tiles_y", 1))
        )

        player_id = self._infer_player_id()
        hand = list(u.battle.players[player_id].hand) if hasattr(u, "battle") else []

        # --- NOOP / WAIT ---
        # If your env has a true NOOP action id, return it here instead of 0.
        if a == -1 or len(hand) == 0:
            return 0

        # --- a is HAND SLOT ---
        card_idx = max(0, min(len(hand) - 1, a))
        env_card_name = hand[card_idx]

        # choose a tile for this specific env card
        tile_index = self._choose_tile_index(env_card_name)

        return int(card_idx * actions_per_tile + tile_index)

    def postprocess_reward(self, info: dict):
        # Keep behavior consistent with other wrappers. If you later want reward shaping, do it here.
        return info.get("reward", 0.0) if isinstance(info, dict) else 0.0

    def _get_wait_token_id(self) -> int:
        # no-op sentinel (postprocess already supports -1)
        return -1

    @torch.no_grad()
    def predict(self, observation: Any) -> int:
        if self.model is None:
            raise ValueError("BC model is not loaded. Call load_model() first.")

        x = observation

        hc = x["history_cards"]
        print(
            "history_cards shape:", tuple(hc.shape),
            "unique:", torch.unique(hc).detach().cpu().tolist()[:20],
            "nonpad:", int((hc != self.pad_id).sum().item()),
            "pad_id:", self.pad_id,
        )

        if isinstance(observation, dict) and "history_cards" in observation:
            if not (isinstance(observation.get("history_cards"), torch.Tensor) and observation["history_cards"].ndim == 2):
                x = self.preprocess_observation(observation)

        hc = x["history_cards"]

        unique_vals = torch.unique(hc)

        if (unique_vals != self.pad_id).any():
            print("unique:", unique_vals.detach().cpu().tolist())

        hand_token_ids = x["deck"][0].to(dtype=torch.long)  # (4,)
        self._last_hand_token_ids = hand_token_ids.detach().cpu().tolist()

        # --- 1) Gate: WAIT vs PLAY ---
        if hasattr(self.model, "forward_gate"):
            gate_logits = self.model.forward_gate(
                x["history_cards"], x["history_players"], x["deck"], x["opp_deck"]
            )  # (B,2)

            gate = int(torch.argmax(gate_logits, dim=-1).item())  # 0=WAIT, 1=PLAY (assumed)

            # DEBUG (leave for now)
            g = gate_logits[0].detach().cpu().tolist()
            print("GATE logits:", g, "gate argmax:", gate)

            probs = torch.softmax(gate_logits, dim=-1)[0]

            print("GATE logits:", gate_logits[0].detach().cpu().tolist(),
                  "probs:", probs.detach().cpu().tolist(),
                  "pad_id:", self.pad_id,
                  "nonpad:", int((x["history_cards"] != self.pad_id).sum().item()))
            
            if gate == 0:
                return -1  # NOOP

        print("made it to a playing checkpoint")
        
        # --- 2) Slot head: choose which card in hand ---
        logits = self.model(
            x["history_cards"], x["history_players"], x["deck"], x["opp_deck"]
        )  # (B,9) in your case

        K = int(logits.shape[-1])
        if K != 9:
            # fallback: if this ever changes
            slot = int(torch.argmax(logits, dim=-1).item())
            return max(0, min(3, slot))

        scores = logits[0]  # shape (9,)
        wait_slot = 8

        # If the model wants to NOOP, OVERRIDE and pick best among 0..3
        slot = int(torch.argmax(scores).item())
        if slot == wait_slot:
            slot = int(torch.argmax(scores[:4]).item())  # best playable slot

        # Clamp to 0..3
        slot = max(0, min(3, slot))

        # DEBUG
        print("SLOT logits:", scores.detach().cpu().tolist(), "chosen slot:", slot, "hand tokens:", self._last_hand_token_ids)

        return slot
