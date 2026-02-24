from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple

import json
import numpy as np
import torch

from src.clasher.model import InferenceModel
from src.clasher.model_state import BCState

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
    BC wrapper (Option 1):
      - BCState is external + owns all history/gating/encode/decode
      - predict(observation, state) reads state (no mutation)
      - postprocess_reward(info, state) is the ONLY mutation point and returns state
      - postprocess_action(model_output, state) delegates to state.decode_action (read-only)
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
    # helpers (kept for compatibility)
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

    def _encode_cards_to_ids(self, cards) -> torch.Tensor:
        """
        Kept because load_model uses it / older code paths might call it,
        but your new state-driven path should NOT need it.
        """
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

    def reset(self, state: Optional[BCState] = None):
        """Reset env and (optionally) reset the external BCState."""
        if state is not None:
            state.reset()
            # allow opening move; set False if you want strictly reactive
            state.should_decide = True
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

        # Build model
        self.model = BCTransformer(vocab_size=vocab_size)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

        if self.printLogs:
            print(f"[BCInferenceModel] Loaded weights: {model_path}")
            print(f"[BCInferenceModel] vocab_size={vocab_size} pad_id={self.pad_id} device={self.device}")

    # --------------------------
    # New state-driven BC API
    # --------------------------
    def update_history_from_info(self, info: dict, state: BCState) -> BCState:
        """Compatibility shim: delegates to BCState.update_from_info (mutation + returns state)."""
        my_id = self._infer_player_id()
        return state.update_from_info(
            info=info,
            env_to_model=ENV_TO_MODEL,
            pad_xy=(PAD_X, PAD_Y),
            my_id=my_id,
            extract_xy_fn=self._extract_xy_from_action,
            printLogs=self.printLogs,
        )

    def preprocess_observation(self, observation: Any, state: BCState) -> Dict[str, torch.Tensor]:
        """Build model inputs from env + BCState (read-only)."""
        if self.token2id is None:
            raise ValueError("token2id not loaded; set bc_args.token2id_path.")
        if self.pad_id is None:
            self.pad_id = int(self.token2id.get("<PAD>", 0))

        return state.encode_inputs(
            env=self.env,
            token2id=self.token2id,
            pad_id=int(self.pad_id),
            env_to_model=ENV_TO_MODEL,
            history_len=int(self.bc_args.history_len),
            device=self.device,
            infer_player_id_fn=self._infer_player_id,
            pad_xy=(PAD_X, PAD_Y),
        )

    @torch.no_grad()
    def predict(self, observation: Any, state: BCState) -> Tuple[int, int, int, int]:
        """Run BC policy forward pass. Reads BCState but does not mutate it."""
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
        if not (
            isinstance(x, dict)
            and "history_cards" in x
            and isinstance(x["history_cards"], torch.Tensor)
            and x["history_cards"].ndim == 2
        ):
            x = self.preprocess_observation(observation, state)

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

    def postprocess_action(self, model_output: Any, state: BCState) -> int:
        """Decode model output into env action using BCState (read-only)."""
        return state.decode_action(
            model_output=model_output,
            env=self.env,
            infer_player_id_fn=self._infer_player_id,
            x_bins=X_BINS,
            y_bins=Y_BINS,
        )

    def postprocess_reward(self, info: dict, state: BCState) -> BCState:
        """ONLY mutation point: update state from env info and return it."""
        return self.update_history_from_info(info, state)
    