import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

def init_lstm(shape):
    shape_fixed = tuple(1 if isinstance(x, str) else x for x in shape)
    zero = np.zeros(shape_fixed, dtype=np.float32)
    return zero, zero

@dataclass
class State:
    def reset(self):
        """Default no-op reset; override in subclasses."""
        return

@dataclass
class ONNXRPPOState(State):
    LSTM_SHAPE = (1, 1, 256)

    pi_h: np.ndarray = field(default_factory=lambda: init_lstm(ONNXRPPOState.LSTM_SHAPE)[0])
    pi_c: np.ndarray = field(default_factory=lambda: init_lstm(ONNXRPPOState.LSTM_SHAPE)[1])
    vf_h: np.ndarray = field(default_factory=lambda: init_lstm(ONNXRPPOState.LSTM_SHAPE)[0])
    vf_c: np.ndarray = field(default_factory=lambda: init_lstm(ONNXRPPOState.LSTM_SHAPE)[1])

    def reset(self):
        self.pi_h, self.pi_c = init_lstm(self.LSTM_SHAPE)
        self.vf_h, self.vf_c = init_lstm(self.LSTM_SHAPE)

@dataclass
class ReplayState(State):
    tick: int = field(default=0)

    def reset(self):
        self.tick = 0

@dataclass
class BCState(State):
    """
    Per-environment BC state.
    Stores history + reactive gating flag.
    Handles encoding model inputs and decoding model outputs.

    IMPORTANT: Per your requirement, the only state mutation should happen
    through update_from_info() (called from postprocess_reward).
    encode_inputs() and decode_action() are read-only.
    """
    history_len: int = 20

    # History uses MODEL tokens (e.g., "hog-rider"), not env names.
    hist_cards: List[str] = field(default_factory=list)
    hist_players: List[int] = field(default_factory=list)
    hist_x: List[int] = field(default_factory=list)
    hist_y: List[int] = field(default_factory=list)

    # Reactive trigger: only act after opponent successfully acts
    should_decide: bool = False

    def reset(self):
        self.hist_cards.clear()
        self.hist_players.clear()
        self.hist_x.clear()
        self.hist_y.clear()
        self.should_decide = False

    def _trim(self):
        H = int(self.history_len)
        if H <= 0:
            self.hist_cards.clear()
            self.hist_players.clear()
            self.hist_x.clear()
            self.hist_y.clear()
            return

        extra = len(self.hist_cards) - H
        if extra > 0:
            del self.hist_cards[:extra]
            del self.hist_players[:extra]
            del self.hist_x[:extra]
            del self.hist_y[:extra]

    # -------------------------
    # READ-ONLY helpers (no mutation)
    # -------------------------
    def _get_player_card_list(self, env, pid: int) -> List[Any]:
        """
        Best-effort card list getter. Tries common attrs.
        Returns a list of env card names or objects already used elsewhere in repo.
        """
        u = getattr(env, "unwrapped", env)
        if not hasattr(u, "battle"):
            return []

        player = u.battle.players[pid]

        for attr in ("deck", "cards", "deck_cards", "full_deck"):
            if hasattr(player, attr):
                v = getattr(player, attr)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return list(v)

        # fallback: some envs only expose current hand
        if hasattr(player, "hand"):
            v = getattr(player, "hand")
            if isinstance(v, (list, tuple)):
                return list(v)

        return []

    def _encode_cards_to_ids(self, cards: List[Any], token2id: Dict[str, int], pad_id: int, env_to_model: Dict[str, str]) -> torch.Tensor:
        # already numeric?
        if isinstance(cards, torch.Tensor):
            if cards.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
                return cards.view(-1).long()

        unk = token2id.get("<UNK>", int(pad_id))
        ids: List[int] = []
        for c in cards:
            if c is None:
                ids.append(int(pad_id))
                continue

            # string env card name -> model token
            if isinstance(c, str):
                model_tok = env_to_model.get(c, c)
                ids.append(int(token2id.get(model_tok, unk)))
            else:
                # if env already gives numeric ids
                try:
                    ids.append(int(c))
                except Exception:
                    ids.append(int(pad_id))

        return torch.tensor(ids, dtype=torch.long).view(-1)

    # -------------------------
    # Encode inputs (READ-ONLY)
    # -------------------------
    def encode_inputs(
        self,
        env,
        token2id: Dict[str, int],
        pad_id: int,
        env_to_model: Dict[str, str],
        history_len: int,
        device,
        infer_player_id_fn,
        pad_xy: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Builds tensors for the BCTransformer forward pass.

        READ-ONLY: does not mutate state. Uses current state history and current env decks.
        """
        pid = int(infer_player_id_fn())
        opp = 1 - pid
        pad_x, pad_y = pad_xy

        # deck + opp_deck from env (current view)
        deck = self._get_player_card_list(env, pid)[:8]
        opp_deck = self._get_player_card_list(env, opp)[:8]

        # pad to length 8
        if len(deck) < 8:
            deck = deck + [None] * (8 - len(deck))
        if len(opp_deck) < 8:
            opp_deck = opp_deck + [None] * (8 - len(opp_deck))

        # history from state (MODEL TOKENS already)
        h_cards = list(self.hist_cards)
        h_players = list(self.hist_players)
        h_x = list(self.hist_x)
        h_y = list(self.hist_y)

        # encode
        history_cards = self._encode_cards_to_ids(h_cards, token2id, pad_id, env_to_model)
        deck_ids = self._encode_cards_to_ids(deck, token2id, pad_id, env_to_model)
        opp_deck_ids = self._encode_cards_to_ids(opp_deck, token2id, pad_id, env_to_model)

        def to_long_1d(x) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.view(-1).long()
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).view(-1).long()
            return torch.tensor(x, dtype=torch.long).view(-1)

        history_players = torch.zeros_like(history_cards) if len(h_players) == 0 else to_long_1d(h_players)
        history_x = torch.full((history_cards.numel(),), int(pad_x), dtype=torch.long) if len(h_x) == 0 else to_long_1d(h_x)
        history_y = torch.full((history_cards.numel(),), int(pad_y), dtype=torch.long) if len(h_y) == 0 else to_long_1d(h_y)

        H = int(history_len)

        # left-pad history to fixed length H
        def _pad_left(t: torch.Tensor, pad_value: int) -> torch.Tensor:
            t = t.view(-1).long()
            if t.numel() > H:
                return t[-H:]
            if t.numel() < H:
                pad_n = H - t.numel()
                return torch.cat([torch.full((pad_n,), int(pad_value), dtype=torch.long), t], dim=0)
            return t

        history_cards = _pad_left(history_cards, int(pad_id))
        history_players = _pad_left(history_players, 0)
        history_x = _pad_left(history_x, int(pad_x))
        history_y = _pad_left(history_y, int(pad_y))

        # pad decks to len 8 (right pad already via None, but make sure tensor len=8)
        def _pad_to_len(t: torch.Tensor, L: int, pad_value: int) -> torch.Tensor:
            t = t.view(-1).long()
            if t.numel() > L:
                return t[:L]
            if t.numel() < L:
                return torch.cat([t, torch.full((L - t.numel(),), int(pad_value), dtype=torch.long)], dim=0)
            return t

        deck_ids = _pad_to_len(deck_ids, 8, int(pad_id))
        opp_deck_ids = _pad_to_len(opp_deck_ids, 8, int(pad_id))

        return {
            "history_cards": history_cards.unsqueeze(0).to(device),
            "history_players": history_players.unsqueeze(0).to(device),
            "history_x": history_x.unsqueeze(0).to(device),
            "history_y": history_y.unsqueeze(0).to(device),
            "deck": deck_ids.view(1, -1).to(device),
            "opp_deck": opp_deck_ids.view(1, -1).to(device),
        }

    # -------------------------
    # Decode action (READ-ONLY)
    # -------------------------
    def decode_action(
        self,
        model_output: Any,
        env,
        infer_player_id_fn,
        x_bins: int,
        y_bins: int,
    ) -> int:
        """
        Converts (gate, deck_idx, x_bin, y_bin) -> env action int.

        READ-ONLY with respect to history. (It DOES consume should_decide? NO.)
        Per your requirement, should_decide consumption should happen in reward update
        logic, not here. So decode_action will only *check* should_decide.
        """
        if not self.should_decide:
            return -1

        if not isinstance(model_output, (tuple, list)) or len(model_output) != 4:
            return -1

        gate, deck_idx, x_bin, y_bin = model_output
        gate = int(gate)
        deck_idx = int(deck_idx)
        x_bin = int(x_bin)
        y_bin = int(y_bin)

        if gate == 0:
            return -1
        if deck_idx == 8:
            return -1

        u = getattr(env, "unwrapped", env)
        pid = int(infer_player_id_fn())

        # current hand from env
        if not hasattr(u, "battle"):
            return -1

        player = u.battle.players[pid]
        hand = list(getattr(player, "hand", []))
        if len(hand) == 0:
            return -1

        # deck list used for deck_idx mapping (must match encode_inputs)
        deck_env_names = self._get_player_card_list(env, pid)[:8]
        if len(deck_env_names) < 8:
            deck_env_names += [None] * (8 - len(deck_env_names))

        env_card_name = deck_env_names[deck_idx] if (0 <= deck_idx < len(deck_env_names)) else None

        # env expects HAND SLOT index
        if env_card_name is not None and env_card_name in hand:
            card_idx = int(hand.index(env_card_name))
        else:
            card_idx = 0

        # bins -> env tile
        tiles_x = int(getattr(u, "tiles_x", x_bins))
        tiles_y = int(getattr(u, "tiles_y", y_bins))

        ex = int(round(x_bin * (tiles_x - 1) / max(1, x_bins - 1))) if tiles_x > 1 else 0
        ey = int(round(y_bin * (tiles_y - 1) / max(1, y_bins - 1))) if tiles_y > 1 else 0
        ex = max(0, min(tiles_x - 1, ex))
        ey = max(0, min(tiles_y - 1, ey))

        tile_index = int(ey * tiles_x + ex)

        actions_per_tile = int(getattr(u, "actions_per_tile", tiles_x * tiles_y))
        tile_index = max(0, min(actions_per_tile - 1, tile_index))

        return int(card_idx * actions_per_tile + tile_index)

    # -------------------------
    # Update from info (MUTATES state) - called from postprocess_reward
    # -------------------------
    def update_from_info(
        self,
        info: Dict[str, Any],
        env_to_model: Dict[str, str],
        pad_xy: Tuple[int, int],
        my_id: int,
        extract_xy_fn,
        printLogs: bool = False,
    ) -> "BCState":
        """
        This is the ONLY mutation point.
        Called from wrapper.postprocess_reward(info, state) and returns self.
        """
        if not isinstance(info, dict):
            return self

        last = info.get("last_action", None)
        if not isinstance(last, dict):
            return self

        opp_id = 1 - int(my_id)
        pad_x, pad_y = pad_xy

        def _read_player(pid: int):
            key = f"player_{pid}"
            action = last.get(key, {})
            if not isinstance(action, dict):
                return False, None, (pad_x, pad_y)
            success = bool(action.get("success", False))
            card_name = action.get("card_name", None)
            xy = extract_xy_fn(action)
            return success, card_name, xy

        # Append both players' successful actions
        for pid in (0, 1):
            success, card_name, (hx, hy) = _read_player(pid)
            if success and card_name and card_name != "None":
                model_card = env_to_model.get(card_name, None)
                if model_card is None:
                    if printLogs:
                        print("UNKNOWN ENV CARD:", repr(card_name), "not in ENV_TO_MODEL")
                    continue
                self.hist_cards.append(model_card)
                self.hist_players.append(int(pid))
                self.hist_x.append(int(hx))
                self.hist_y.append(int(hy))

        # Reactive trigger: only decide after opponent moves successfully
        opp_success, opp_card, _ = _read_player(opp_id)
        if opp_success and opp_card and opp_card != "None":
            self.should_decide = True
        else:
            # consume trigger when we tried a step without opponent move
            # (keeps behavior stable in fast loops)
            self.should_decide = False

        self._trim()
        return self
    