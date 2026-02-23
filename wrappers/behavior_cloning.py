from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple

import json
import numpy as np
import torch

from src.clasher.model import InferenceModel

# Prefer new location; fallback for older repo layouts
try:
    from model import BCTransformer
except Exception:  # pragma: no cover
    from bc_transformer.train.model import BCTransformer


# ----------------------------
# Card name mapping
# bc_model token -> gym env card name
# ----------------------------
MODEL_TO_ENV = {
    "cannon": "Cannon",
    "fireball": "Fireball",
    "hog-rider": "HogRider",
    "ice-golem": "IceGolemite",
    "ice-spirit": "IceSpirits",
    "musketeer": "Musketeer",
    "skeletons": "Skeletons",
    "the-log": "Log",
}
ENV_TO_MODEL = {v: k for k, v in MODEL_TO_ENV.items()}

# Must match train.py constants / model expectations
X_BINS = 18  # x_tile: 0..17
Y_BINS = 32  # y_tile: 0..31
PAD_X = 18
PAD_Y = 32


@dataclass
class BCArgs:
    token2id_path: Optional[str] = None
    pad_id: Optional[int] = None
    device: str = "cpu"
    history_len: int = 20

    # Optional explicit weights path (if inference provides it)
    model_path: Optional[str] = None

    # Whether the wrapper should try to auto-discover weights if model_path is missing.
    autoload_weights: bool = True


class BCInferenceModel(InferenceModel):
    """
    Updated BC wrapper:
      - loads token2id in __init__ (so preprocess works)
      - loads weights from bc_args.model_path OR auto-discovers (prefers weight_history)
      - reactive gating: only decide when opponent successfully acts
      - model outputs: gate, card, x, y
    """

    def __init__(self, env=None, bc_args: Optional[BCArgs] = None, printLogs: bool = False):
        super().__init__()

        self.env = env
        self.printLogs = printLogs
        self.bc_args = bc_args or BCArgs()

        self.model: Optional[BCTransformer] = None
        self.token2id: Optional[Dict[str, int]] = None
        self.pad_id: Optional[int] = self.bc_args.pad_id

        dev = self.bc_args.device
        self.device = torch.device(dev if (dev.startswith("cuda") and torch.cuda.is_available()) else "cpu")

        # buffers
        self._hist_cards = None
        self._hist_players = None
        self._hist_x = None
        self._hist_y = None

        # reactive trigger: only decide after opponent makes a successful move
        self._should_decide = False  # allow opening move by default (change to False if you want strictly reactive)
        self._last_deck_env_names = [None] * 8

        # 1) Load token2id now (so preprocess works even before weights load)
        t2i_path = getattr(self.bc_args, "token2id_path", None)
        if t2i_path:
            t2i_path = Path(t2i_path)
            if not t2i_path.exists():
                raise FileNotFoundError(f"token2id_path does not exist: {t2i_path}")
            with open(t2i_path, "r") as f:
                self.token2id = json.load(f)
            if self.pad_id is None:
                self.pad_id = int(self.token2id.get("<PAD>", 0))
            if self.printLogs:
                print(f"[BCInferenceModel] Loaded token2id from: {t2i_path} | pad_id={self.pad_id}")
        else:
            if self.printLogs:
                print("[BCInferenceModel] WARNING: bc_args.token2id_path is None; token2id not loaded yet.")

        # 2) Load weights if provided explicitly, else (optionally) auto-discover.
        mp = getattr(self.bc_args, "model_path", None)
        if mp:
            self.load_model(mp)
        else:
            if getattr(self.bc_args, "autoload_weights", True):
                self._try_autoload_weights_from_vocab_folder()

    # ----------------------------
    # helpers
    # ----------------------------
    def _env_card_to_model_token(self, env_card: str) -> str:
        return ENV_TO_MODEL.get(env_card, env_card)

    def _model_token_to_env_card(self, model_token: str) -> str:
        return MODEL_TO_ENV.get(model_token, model_token)

    def _infer_player_id(self) -> int:
        u = getattr(self.env, "unwrapped", self.env)
        if hasattr(u, "opponent_policy") and getattr(u, "opponent_policy") is self:
            return 1
        return 0

    def _ensure_hist_buffers(self):
        if self._hist_cards is None:
            from collections import deque
            H = int(self.bc_args.history_len)
            self._hist_cards = deque(maxlen=H)
            self._hist_players = deque(maxlen=H)
            self._hist_x = deque(maxlen=H)
            self._hist_y = deque(maxlen=H)

    def _encode_cards_to_ids(self, cards) -> torch.Tensor:
        if self.token2id is None:
            raise ValueError("token2id not loaded (need token2id.json or embedded in checkpoint).")

        # already numeric?
        if isinstance(cards, torch.Tensor):
            if cards.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
                return cards.view(-1).long()
        if isinstance(cards, np.ndarray) and np.issubdtype(cards.dtype, np.integer):
            return torch.from_numpy(cards).view(-1).long()
        if isinstance(cards, list) and (len(cards) == 0 or isinstance(cards[0], (int, np.integer))):
            return torch.tensor(cards, dtype=torch.long).view(-1)

        unk = self.token2id.get("<UNK>", int(self.pad_id if self.pad_id is not None else 0))
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

    def _extract_xy_from_action(self, action: Dict[str, Any]) -> Tuple[int, int]:
        if not isinstance(action, dict):
            return PAD_X, PAD_Y

        for kx, ky in (("x", "y"), ("tile_x", "tile_y")):
            if kx in action and ky in action:
                try:
                    x = int(action.get(kx))
                    y = int(action.get(ky))
                    x = max(0, min(X_BINS - 1, x))
                    y = max(0, min(Y_BINS - 1, y))
                    return x, y
                except Exception:
                    pass

        for kt in ("tile_index", "tile", "placement"):
            if kt in action:
                v = action.get(kt)
                try:
                    tile_index = int(v)
                except Exception:
                    continue

                u = getattr(self.env, "unwrapped", self.env)
                tiles_x = int(getattr(u, "tiles_x", X_BINS))
                tiles_y = int(getattr(u, "tiles_y", Y_BINS))
                ex = tile_index % tiles_x
                ey = tile_index // tiles_x

                mx = int(round(ex * (X_BINS - 1) / max(1, tiles_x - 1))) if tiles_x > 1 else 0
                my = int(round(ey * (Y_BINS - 1) / max(1, tiles_y - 1))) if tiles_y > 1 else 0
                mx = max(0, min(X_BINS - 1, mx))
                my = max(0, min(Y_BINS - 1, my))
                return mx, my

        return PAD_X, PAD_Y

    def _looks_like_new_model_state_dict(self, sd: Dict[str, torch.Tensor]) -> bool:
        # New model has these keys
        needed = ("x_emb.weight", "y_emb.weight", "x_head.weight", "y_head.weight")
        return all(k in sd for k in needed)

    def _try_autoload_weights_from_vocab_folder(self):
        t2i_path = getattr(self.bc_args, "token2id_path", None)
        if not t2i_path:
            return

        base = Path(t2i_path).parent

        # IMPORTANT: prefer weight_history first (your new checkpoints)
        weight_candidates = [
            base / "weight_history" / "model_state_dict_v1.pt",
            base / "weight_history" / "model_state_dict.pt",
            base / "weight_history" / "checkpoint.pt",
            base / "weight_history" / "model.pt",

            base / "model_state_dict_v1.pt",
            base / "model_state_dict.pt",
            base / "checkpoint.pt",
            base / "model.pt",
            base / "bc_model.pt",
        ]

        for c in weight_candidates:
            if c.exists():
                # If multiple exist, prefer the one that matches new arch.
                try:
                    ckpt = torch.load(str(c), map_location="cpu")
                    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                    if isinstance(sd, dict) and self._looks_like_new_model_state_dict(sd):
                        if self.printLogs:
                            print(f"[BCInferenceModel] Auto-loading NEW-model weights: {c}")
                        self.load_model(c)
                        return
                except Exception:
                    pass

        # fallback: load first existing candidate even if we couldn't inspect
        for c in weight_candidates:
            if c.exists():
                if self.printLogs:
                    print(f"[BCInferenceModel] Auto-loading weights (fallback): {c}")
                self.load_model(c)
                return

        if self.printLogs:
            print("[BCInferenceModel] No weights auto-loaded (none found).")

    # --------------------------
    # Interface required by inference.py
    # --------------------------
    def wrap_env(self, env):
        self.env = env
        return env

    def reset(self):
        # allow opening move; if you want STRICT reactive, set to False here.
        self._should_decide = True
        return self.env.reset()

    def load_model(self, model_path: Union[str, Path]):
        model_path = Path(model_path)

        ckpt = torch.load(str(model_path), map_location=self.device)
        state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt

        # Ensure token2id exists (prefer already-loaded; else load from path / sidecar)
        if self.token2id is None:
            t2i_path = getattr(self.bc_args, "token2id_path", None)
            if t2i_path:
                with open(t2i_path, "r") as f:
                    self.token2id = json.load(f)

        if self.token2id is None and isinstance(ckpt, dict):
            for k in ("token2id", "token_to_id", "vocab", "vocab_dict"):
                if k in ckpt and isinstance(ckpt[k], dict) and len(ckpt[k]) > 0:
                    self.token2id = ckpt[k]
                    break

        if self.token2id is None:
            sidecar = model_path.parent / "token2id.json"
            if sidecar.exists():
                with open(sidecar, "r") as f:
                    self.token2id = json.load(f)

        if self.token2id is None:
            raise ValueError(
                "token2id not loaded. Provide bc_args.token2id_path, "
                "or place token2id.json next to weights, or embed token2id in checkpoint."
            )

        self.token2id = {str(k): int(v) for k, v in self.token2id.items()}
        if self.pad_id is None:
            self.pad_id = int(self.token2id.get("<PAD>", 0))

        # Infer vocab size from checkpoint embedding (preferred)
        emb_key = None
        if isinstance(state, dict):
            if "tok_emb.weight" in state:
                emb_key = "tok_emb.weight"
            elif "model.tok_emb.weight" in state:
                emb_key = "model.tok_emb.weight"

        vocab_size = int(state[emb_key].shape[0]) if emb_key is not None else len(self.token2id)

        # Pad token2id to cover ids if needed
        if len(self.token2id) < vocab_size:
            used_ids = set(self.token2id.values())
            for tid in range(vocab_size):
                if tid in used_ids:
                    continue
                name = f"<EXTRA_{tid}>"
                while name in self.token2id:
                    name += "_"
                self.token2id[name] = tid
                used_ids.add(tid)

        m = BCTransformer(
            vocab_size=vocab_size,
            pad_id=int(self.pad_id),
            n_actions=9,  # 8 deck slots + NOOP
        ).to(self.device)

        m.load_state_dict(state, strict=True)
        m.eval()
        self.model = m

        if self.printLogs:
            print(f"[BCInferenceModel] Loaded weights: {model_path}")
            print(f"[BCInferenceModel] vocab_size={vocab_size} pad_id={self.pad_id} device={self.device}")

    def update_history_from_info(self, info: dict):
        self._ensure_hist_buffers()

        if not isinstance(info, dict):
            return

        last = info.get("last_action", None)
        if not isinstance(last, dict):
            return

        my_id = self._infer_player_id()
        opp_id = 1 - my_id

        # Helper to read "player_0"/"player_1"
        def _read_player(player_id: int) -> Tuple[bool, Optional[str], Tuple[int, int]]:
            key = f"player_{player_id}"
            action = last.get(key, {})
            if not isinstance(action, dict):
                return False, None, (PAD_X, PAD_Y)

            success = bool(action.get("success", False))
            card_name = action.get("card_name", None)
            xy = self._extract_xy_from_action(action)
            return success, card_name, xy

        # append BOTH players' successful actions to history (training usually did this)
        for pid in (0, 1):
            success, card_name, (hx, hy) = _read_player(pid)
            if success and card_name and card_name != "None":
                model_card = ENV_TO_MODEL.get(card_name, None)
                if model_card is None:
                    if self.printLogs:
                        print("UNKNOWN ENV CARD:", repr(card_name), "not in ENV_TO_MODEL")
                    continue

                self._hist_cards.append(model_card)
                self._hist_players.append(pid)
                self._hist_x.append(hx)
                self._hist_y.append(hy)

        # ✅ reactive trigger: only decide after opponent moves successfully
        opp_success, opp_card, _ = _read_player(opp_id)
        if opp_success and opp_card and opp_card != "None":
            self._should_decide = True

    def preprocess_observation(self, observation: Any) -> Dict[str, torch.Tensor]:
        self._ensure_hist_buffers()

        if self.token2id is None:
            raise ValueError("token2id not loaded; set bc_args.token2id_path.")

        u = getattr(self.env, "unwrapped", self.env)

        # Try structured obs from env
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

        if obs_dict is None:
            if not hasattr(u, "battle"):
                raise TypeError(
                    "Env obs is pixels, and env doesn't expose battle state or get_bc_obs(). "
                    "Add env.get_bc_obs() or expose u.battle."
                )

            pid = self._infer_player_id()
            opp = 1 - pid

            # capture deck names for mapping deck_idx -> env card name later
            player_obj = u.battle.players[pid]
            deck_list = None
            for attr in ("deck", "cards", "deck_cards", "full_deck"):
                if hasattr(player_obj, attr):
                    v = getattr(player_obj, attr)
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        deck_list = list(v)
                        break
            if deck_list is None and hasattr(player_obj, "hand"):
                deck_list = list(player_obj.hand)
            if deck_list is None:
                deck_list = []
            self._last_deck_env_names = [c for c in deck_list][:8]
            if len(self._last_deck_env_names) < 8:
                self._last_deck_env_names += [None] * (8 - len(self._last_deck_env_names))

            def _get_8_cards(player_obj2):
                for attr2 in ("deck", "cards", "deck_cards", "full_deck"):
                    if hasattr(player_obj2, attr2):
                        v2 = getattr(player_obj2, attr2)
                        if isinstance(v2, (list, tuple)) and len(v2) > 0:
                            return list(v2)
                if hasattr(player_obj2, "hand"):
                    return list(player_obj2.hand)
                return []

            p0_cards = _get_8_cards(u.battle.players[pid])
            p1_cards = _get_8_cards(u.battle.players[opp])

            obs_dict = {
                "history_cards": list(self._hist_cards),
                "history_players": list(self._hist_players),
                "history_x": list(self._hist_x),
                "history_y": list(self._hist_y),
                "deck": p0_cards,
                "opp_deck": p1_cards,
            }

        h_cards = obs_dict.get("history_cards", [])
        h_players = obs_dict.get("history_players", [])
        h_x = obs_dict.get("history_x", [])
        h_y = obs_dict.get("history_y", [])
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

        history_cards = self._encode_cards_to_ids(h_cards if h_cards is not None else [])
        deck_ids = self._encode_cards_to_ids(deck if deck is not None else [])
        opp_deck_ids = self._encode_cards_to_ids(opp_deck if opp_deck is not None else [])

        history_players = torch.zeros_like(history_cards) if (h_players is None or len(h_players) == 0) else to_long_1d(h_players)
        history_x = torch.full((history_cards.numel(),), PAD_X, dtype=torch.long) if (h_x is None or len(h_x) == 0) else to_long_1d(h_x)
        history_y = torch.full((history_cards.numel(),), PAD_Y, dtype=torch.long) if (h_y is None or len(h_y) == 0) else to_long_1d(h_y)

        H = int(self.bc_args.history_len)

        def _pad_left(t: torch.Tensor, pad_value: int) -> torch.Tensor:
            if t.numel() > H:
                return t[-H:]
            if t.numel() < H:
                pad_n = H - t.numel()
                return torch.cat([torch.full((pad_n,), int(pad_value), dtype=torch.long), t], dim=0)
            return t

        history_cards = _pad_left(history_cards, int(self.pad_id))
        history_players = _pad_left(history_players, 0)
        history_x = _pad_left(history_x, PAD_X)
        history_y = _pad_left(history_y, PAD_Y)

        def _pad_to_len(t: torch.Tensor, L: int, pad_value: int) -> torch.Tensor:
            t = t.view(-1).long()
            if t.numel() > L:
                return t[:L]
            if t.numel() < L:
                return torch.cat([t, torch.full((L - t.numel(),), int(pad_value), dtype=torch.long)], dim=0)
            return t

        deck_ids = _pad_to_len(deck_ids, 8, int(self.pad_id))
        opp_deck_ids = _pad_to_len(opp_deck_ids, 8, int(self.pad_id))

        return {
            "history_cards": history_cards.unsqueeze(0).to(self.device),
            "history_players": history_players.unsqueeze(0).to(self.device),
            "history_x": history_x.unsqueeze(0).to(self.device),
            "history_y": history_y.unsqueeze(0).to(self.device),
            "deck": deck_ids.view(1, -1).to(self.device),
            "opp_deck": opp_deck_ids.view(1, -1).to(self.device),
        }

    @torch.no_grad()
    def predict(self, observation: Any) -> Tuple[int, int, int, int]:
        # If inference.py forgot to load, try auto-discovery once more.
        if self.model is None:
            mp = getattr(self.bc_args, "model_path", None)
            if mp:
                self.load_model(mp)
            elif getattr(self.bc_args, "autoload_weights", True):
                self._try_autoload_weights_from_vocab_folder()

        if self.model is None:
            raise ValueError("BC model is not loaded. Call load_model() first or set bc_args.model_path.")

        x = observation
        if not (isinstance(x, dict) and "history_cards" in x and isinstance(x["history_cards"], torch.Tensor) and x["history_cards"].ndim == 2):
            x = self.preprocess_observation(observation)

        gate_logits, card_logits, x_logits, y_logits = self.model.forward_policy(
            history_cards=x["history_cards"],
            history_players=x["history_players"],
            deck=x["deck"],
            opp_deck=x["opp_deck"],
            history_x=x["history_x"],
            history_y=x["history_y"],
        )

        gate = int(torch.argmax(gate_logits, dim=-1).item())      # 0=WAIT, 1=PLAY
        deck_idx = int(torch.argmax(card_logits, dim=-1).item())  # 0..7 or 8=NOOP
        x_bin = int(torch.argmax(x_logits, dim=-1).item())        # 0..17
        y_bin = int(torch.argmax(y_logits, dim=-1).item())        # 0..31

        if self.printLogs:
            print("[BC] gate:", gate, "| deck_idx:", deck_idx, "| x_bin:", x_bin, "| y_bin:", y_bin)

        return gate, deck_idx, x_bin, y_bin

    def postprocess_action(self, model_output: Any) -> int:
        # ✅ REACTIVE TRIGGER: only act after opponent moved
        if not self._should_decide:
            return -1

        if not isinstance(model_output, (tuple, list)) or len(model_output) != 4:
            return -1

        gate, deck_idx, x_bin, y_bin = model_output
        gate = int(gate)
        deck_idx = int(deck_idx)
        x_bin = int(x_bin)
        y_bin = int(y_bin)

        # consume trigger whether or not we actually play (prevents spamming)
        self._should_decide = False

        if gate == 0:
            return -1
        if deck_idx == 8:
            return -1

        u = getattr(self.env, "unwrapped", self.env)
        actions_per_tile = int(getattr(u, "actions_per_tile", getattr(u, "tiles_x", 1) * getattr(u, "tiles_y", 1)))

        pid = self._infer_player_id()
        hand = list(u.battle.players[pid].hand) if hasattr(u, "battle") and hasattr(u.battle.players[pid], "hand") else []
        if len(hand) == 0:
            return -1

        # deck_idx -> env card name (best effort)
        env_card_name = None
        if isinstance(self._last_deck_env_names, list) and 0 <= deck_idx < len(self._last_deck_env_names):
            env_card_name = self._last_deck_env_names[deck_idx]

        # env expects HAND SLOT index
        if env_card_name is not None and env_card_name in hand:
            card_idx = int(hand.index(env_card_name))
        else:
            # fallback: pick slot 0 if predicted card not in hand
            card_idx = 0

        # (x_bin,y_bin) -> env tile index
        tiles_x = int(getattr(u, "tiles_x", X_BINS))
        tiles_y = int(getattr(u, "tiles_y", Y_BINS))

        ex = int(round(x_bin * (tiles_x - 1) / max(1, X_BINS - 1))) if tiles_x > 1 else 0
        ey = int(round(y_bin * (tiles_y - 1) / max(1, Y_BINS - 1))) if tiles_y > 1 else 0
        ex = max(0, min(tiles_x - 1, ex))
        ey = max(0, min(tiles_y - 1, ey))

        tile_index = int(ey * tiles_x + ex)
        tile_index = max(0, min(actions_per_tile - 1, tile_index))

        return int(card_idx * actions_per_tile + tile_index)

    def postprocess_reward(self, info: dict):
        self.update_history_from_info(info)
        return info.get("reward", 0.0) if isinstance(info, dict) else 0.0
    