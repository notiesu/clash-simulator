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

from bc_transformer.train.model import BCTransformer


DECK = ["Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons", "Log"]

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

    def __init__(self, env=None, bc_args: Optional[BCArgs] = None, printLogs: bool = False, player_id = 0):
        super().__init__()
        self.player_id = player_id
        self.env = env
        self.printLogs = printLogs
        self.bc_args = bc_args if bc_args is not None else BCArgs()

        self.model: Optional[BCTransformer] = None
        self.token2id: Optional[Dict[str, int]] = None
        self.pad_id: Optional[int] = self.bc_args.pad_id

        self.device = self.bc_args.device

        # 1) Load token2id now (so preprocess works even before weights load)
        t2i_path = getattr(self.bc_args, "token2id_path", None)
        print(t2i_path)
        if t2i_path:
            t2i_path = Path(t2i_path)
            if not t2i_path.exists():
                raise FileNotFoundError(f"token2id_path does not exist: {t2i_path}")
            with open(t2i_path, "r") as f:
                self.token2id = json.load(f)

            if self.pad_id is None:
                self.pad_id = int(self.token2id.get("<PAD>", 0))
            print(f"[BCInferenceModel] Loaded token2id from: {t2i_path} | pad_id={self.pad_id}")
        else:
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
        return self.player_id

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
            # STRICTLY reactive: do NOT allow an opening move
            state.should_decide = False
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
        """
        Compatibility shim: delegates to BCState.update_from_info (mutation + returns state).
    
        IMPORTANT:
          - BCState.update_from_info sets state.should_decide.
          - Do NOT override should_decide here.
        """
        if state is None:
            return state
        if not isinstance(info, dict):
            return state
    
        my_id = int(self._infer_player_id())
    
        # Prefer instance-owned mappings/pads if they exist; fall back to module constants.
        env_to_model = getattr(self, "env_to_model", None)
        if env_to_model is None:
            env_to_model = globals().get("ENV_TO_MODEL", {})
    
        pad_x = getattr(self, "pad_x", None)
        pad_y = getattr(self, "pad_y", None)
        if pad_x is None or pad_y is None:
            pad_x = globals().get("PAD_X", -1)
            pad_y = globals().get("PAD_Y", -1)
    
        # 1) Update history + should_decide inside BCState (source of truth)
        state = state.update_from_info(
            info=info,
            env_to_model=env_to_model,
            pad_xy=(int(pad_x), int(pad_y)),
            my_id=my_id,
            extract_xy_fn=self._extract_xy_from_action,
            printLogs=self.printLogs,
        )
    
        # 2) Optional debug print
        if self.printLogs:
            opp_id = 1 - my_id
            la = info.get("last_action", {})
            opp = la.get(f"player_{opp_id}", {}) if isinstance(la, dict) else {}
            opponent_played = (
                isinstance(opp, dict)
                and bool(opp.get("success", False))
                and opp.get("card_name") not in (None, "None", "")
            )
            print(
                f"[BC gating] my_id={my_id} opponent_played={opponent_played} -> should_decide={state.should_decide}"
            )
    
        return state

    def preprocess_observation(self, observation: Any, state) -> Dict[str, torch.Tensor]:
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
            pid=self.player_id,
            pad_xy=(PAD_X, PAD_Y),
        )

    def _mask_to_current_hand(
    self,
    card_logits: torch.Tensor,
    info,
    *,
    noop_index: int = 8,
) -> tuple[torch.Tensor, list[bool]]:
        """
        Mask deck-slot logits (0..7) so we only select cards that are currently in-hand.
        Keep NOOP index (8) always allowed.

        Returns:
          masked_logits: same shape as card_logits
          allowed: python list[bool] length 9 (0..8)
        """
        hand = info['players'][self.player_id]['hand'] if isinstance(info, dict) else []

        # deck_env_names MUST match how BCState.encode_inputs forms the 8-card deck view
        deck_env_names = DECK
        if len(deck_env_names) < 8:
            deck_env_names += [None] * (8 - len(deck_env_names))

        allowed = [False] * 9
        for i in range(8):
            cn = deck_env_names[i]
            allowed[i] = (cn is not None) and (cn in hand)

        allowed[noop_index] = True  # always allow "no play"

        masked = card_logits.clone()
        # card_logits is shape [B, 9] (or [1,9]) in your setup
        neg_inf = torch.finfo(masked.dtype).min
        mask_tensor = torch.tensor(allowed, device=masked.device, dtype=torch.bool).view(1, -1)
        masked = torch.where(mask_tensor, masked, torch.full_like(masked, neg_inf))
        return masked, allowed

    @torch.no_grad()
    def predict(self, observation: Any, valid_action_mask = None, state=None, info=None):
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

        gate = int(torch.argmax(gate_logits, dim=-1).item())  # 0=WAIT, 1=PLAY

        # NEW: mask deck choices to only current hand
        card_logits_masked, allowed = self._mask_to_current_hand(card_logits, info, noop_index=8)

        deck_idx_raw = int(torch.argmax(card_logits, dim=-1).item())          # 0..7 or 8=NOOP
        deck_idx = int(torch.argmax(card_logits_masked, dim=-1).item())       # 0..7 or 8=NOOP (masked)
        x_bin = int(torch.argmax(x_logits, dim=-1).item())                    # 0..17
        y_bin = int(torch.argmax(y_logits, dim=-1).item())                    # 0..31

        if self.printLogs:
            print(
                f"[BC] gate: {gate} | deck_idx(raw): {deck_idx_raw} | deck_idx(masked): {deck_idx} "
                f"| x_bin: {x_bin} | y_bin: {y_bin} | allowed={allowed}"
            )

        """Decode model output into env action using BCState (read-only)."""

        #get action
        model_output = (gate, deck_idx, x_bin, y_bin)
        action_int = state.decode_action(
            model_output=model_output,
            info=info,
            pid=self.player_id,
            x_bins=X_BINS,
            y_bins=Y_BINS,
        )

        #update state
        next_state = self.update_history_from_info(info, state)
        return action_int, next_state

    def postprocess_action(self, action) -> int:
        return action

    def postprocess_reward(self, info: dict, state: BCState) -> BCState:
        return 0.0
    