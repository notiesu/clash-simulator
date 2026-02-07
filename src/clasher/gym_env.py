"""Gymnasium-compatible environment wrapping the Clash Royale battle engine.

This provides a minimal, working `ClashRoyaleGymEnv` implementation suitable
for running the project's `helloworld.py` demo and for basic RL loops.
"""

# TODO - Benchmark on tick times for latency
import contextlib
from typing import Tuple, Dict, Any, Optional

import json
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
from .engine import BattleEngine
from .arena import TileGrid, Position
from pettingzoo import ParallelEnv
from .model import InferenceModel
from collections import deque

MAX_TOTAL_TOWER_HP = 12086  # 4824 + 2 * 363

class ClashRoyaleGymEnv(gym.Env):
    """Simple Gymnasium environment wrapper around BattleEngine/BattleState.

    Action encoding (discrete 2304): card_idx (4) x x_tile (18) x y_tile (32)
    Observation: dict with `'p1-view'` -> 128x128x3
    Channel 1 - 0 = p0, 1 = p1,
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self,
                 speed_factor: float = 1.0,
                 data_file: str = "gamedata.json",
                 max_steps: int = 9090,
                 suppress_output: bool = True,
                 decks_file: Optional[str] = None,
                 deck0: Optional[list] = None,
                 deck1: Optional[list] = None,
                 deck0_name: Optional[str] = None,
                 deck1_name: Optional[str] = None):
        super().__init__()
        self.speed_factor = speed_factor
        self.data_file = data_file
        self.engine = BattleEngine(self.data_file)

        # Create initial battle instance
        self.battle = self.engine.create_battle()

        # Fixed action space as in README: 4 * 18 * 32 = 2304
        self.num_cards = 4
        self.tiles_x = self.battle.arena.width
        self.tiles_y = self.battle.arena.height
        self.actions_per_tile = self.tiles_x * self.tiles_y

        # Deck configuration options (can be lists of card names or names/indexes referring to decks.json)
        self._decks_file = "decks.json"
        self._initial_deck0 = deck0
        self._initial_deck1 = deck1
        self._initial_deck0_name = deck0_name
        self._initial_deck1_name = deck1_name

        self.no_op_action = self.num_cards * self.actions_per_tile
        self.action_space = spaces.Discrete(self.num_cards * self.actions_per_tile + 1)

        # Observation: 128x128 RGB-like tensor
        self.obs_shape = (128, 128, 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)

        # Internal bookkeeping
        self._step_count = 0
        self._max_steps = max_steps
        self._prev_score = self._compute_score()
        # Mapping from unit type name to compact type id (1..254)
        self._type_to_id: Dict[str, int] = {}
        self._next_type_id = 1

        # Set opponent policy - can embed within the environment step
        # Opponent policy: either None (no embedded opponent) or an InferenceModel instance
        self.opponent_policy: Optional[InferenceModel] = None

        # Apply any initial decks requested by constructor
        try:
            self._apply_initial_decks()
        except Exception:
            # don't fail init on deck errors; caller can set decks later
            pass

    def set_opponent_policy(self, model: InferenceModel):
        """Set an opponent policy model to be used for player 1 actions."""
        self.opponent_policy = model

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            return None
        np.random.seed(seed)
        return seed

    def set_player_deck(self, player_id: int, deck: list, initial_hand: Optional[list] = None):
        """Set the deck for a specific player and rebuild their hand/cycle.

        player_id: 0 or 1
        deck: list of card names (ordered)
        initial_hand: optional list of 4 card names to use as the starting hand
        """
        if player_id < 0 or player_id >= len(self.battle.players):
            raise IndexError("player_id out of range")
        player = self.battle.players[player_id]
        player.deck = list(deck)
        # Set initial hand
        if initial_hand is not None:
            player.hand = list(initial_hand)
        else:
            player.hand = list(deck[:4])
        # Rebuild cycle queue from remaining deck cards
        remaining = [c for c in player.deck if c not in player.hand]
        player.cycle_queue = deque(remaining)

    def get_valid_action_mask(self, player_id: int) -> np.ndarray:
        """Return a boolean mask over the discrete action space for the given player_id."""
        if player_id < 0 or player_id >= len(self.battle.players):
            raise IndexError("player_id out of range")
        player = self.battle.players[player_id]
        arena = self.battle.arena

        mask = np.zeros(self.action_space.n, dtype=bool)
        for card_idx, card_name in enumerate(player.hand):
            card_stats = self.battle.card_loader.get_card(card_name)
            if card_stats is None:
                print(f"Warning: card '{card_name}' not found in card loader")
            if not player.can_play_card(card_name, card_stats):
                continue
            for x in range(self.tiles_x):
                for y in range(self.tiles_y):
                    # y here is the world/tile y; for player 1 the action encoding is
                    # canonicalized (y flipped) so we must compute the encoded tile y.
                    pos = Position(float(x) + 0.5, float(y) + 0.5)
                    if arena.can_deploy_at(pos, player_id, self.battle):
                        encoded_y = y if player_id == 0 else (self.tiles_y - 1 - y)
                        action_int = card_idx * self.actions_per_tile + encoded_y * self.tiles_x + x
                        mask[int(action_int)] = True

        # allow no-op always
        mask[int(self.no_op_action)] = True
        return mask

    def _load_decks_file(self) -> list:
        """Load decks from a JSON file path stored in `self._decks_file`.

        Returns a list of deck dicts with keys like 'name' and 'cards'.
        """
        path = self._decks_file
        if not path:
            return []
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        try:
            with open(path, 'r') as fh:
                data = json.load(fh)
            return data.get('decks', [])
        except Exception:
            return []

    def _apply_initial_decks(self):
        """Apply deck configuration provided at construction time.

        Priority: explicit `deck0`/`deck1` lists > deck names `deck0_name`/`deck1_name` in `decks_file` > first two decks in `decks_file`.
        """
        # If explicit lists provided, use them
        if self._initial_deck0 is not None and self._initial_deck1 is not None:
            self.set_decks(self._initial_deck0, self._initial_deck1)
            return

        # Otherwise, try to load from decks_file
        decks = self._load_decks_file()
        if not decks:
            return

        # Helper to resolve by name or index
        def resolve(name_or_index):
            if name_or_index is None:
                return None
            # integer index
            try:
                idx = int(name_or_index)
                if 0 <= idx < len(decks):
                    return decks[idx].get('cards', [])
            except Exception:
                pass
            # string name lookup
            for d in decks:
                if str(d.get('name', '')).lower() == str(name_or_index).lower():
                    return d.get('cards', [])
            return None

        d0 = resolve(self._initial_deck0_name) if self._initial_deck0_name else None
        d1 = resolve(self._initial_deck1_name) if self._initial_deck1_name else None

        # If neither named decks specified, fall back to first two decks in file
        if d0 is None and d1 is None:
            if len(decks) >= 2:
                self.set_decks(decks[0].get('cards', []), decks[1].get('cards', []))
            elif len(decks) == 1:
                self.set_player_deck(0, decks[0].get('cards', []))
        else:
            if d0 is not None:
                self.set_player_deck(0, d0)
            if d1 is not None:
                self.set_player_deck(1, d1)

    def set_decks(self, deck0: list, deck1: list, hand0: Optional[list] = None, hand1: Optional[list] = None):
        """Convenience to set both players' decks and optional starting hands."""
        self.set_player_deck(0, deck0, initial_hand=hand0)
        self.set_player_deck(1, deck1, initial_hand=hand1)
    
    def transpose_observation(self, observation):
        """
        Transpose observation to switch p0 <-> p1. 
        DO NOT OVERRIDE! PASS IN ENV OUTPUT OBSERVATION, NOT PROCESSED!
        """
        """
        Tranpose p0 and p1 obs
        Rules - Switch p0 and p1
        Switch entities ownership
        Switch x,y -> x, mirror y 
        """

        arr = np.asarray(observation)
        # print(arr)
        trans = arr[::-1, :, :].copy()
        
        owner = trans[..., 0]
        owner_new = owner.copy()
        owner_new[owner == 255] = 128
        owner_new[owner == 128] = 255
        trans[..., 0] = owner_new
        
        return trans
        

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        if seed is not None:
            self.seed(seed)

        # Recreate engine & battle to ensure clean state
        self.engine = BattleEngine(self.data_file)
        self.battle = self.engine.create_battle()
        self._step_count = 0
        self._prev_score = self._compute_score()

        obs = self._render_obs()
        info: Dict[str, Any] = {
            "tick": self.battle.tick,
            "time": self.battle.time,
        }

        # (players meta will be attached below alongside entities)

        # entities meta (same format as step)
        entities_meta = []
        arena = self.battle.arena
        for eid, ent in self.battle.entities.items():
            card_stats = getattr(ent, 'card_stats', None)
            type_name = getattr(card_stats, 'name', None) if card_stats is not None else ent.__class__.__name__
            type_id = self._type_to_id.get(type_name, 0)

            px = int((ent.position.x / arena.width) * self.obs_shape[1])
            py = int((ent.position.y / arena.height) * self.obs_shape[0])
            px = max(0, min(self.obs_shape[1] - 1, px))
            py = max(0, min(self.obs_shape[0] - 1, py))

            try:
                hp_frac = max(0.0, min(1.0, ent.hitpoints / max(1.0, getattr(ent, 'max_hitpoints', 1.0))))
            except Exception:
                hp_frac = 1.0

            entities_meta.append({
                "id": getattr(ent, 'id', None),
                "type_name": type_name,
                "type_id": int(type_id),
                "player_id": getattr(ent, 'player_id', None),
                "position": (float(ent.position.x), float(ent.position.y)),
                "pixel": (int(px), int(py)),
                "hp": float(getattr(ent, 'hitpoints', 0)),
                "hp_frac": float(hp_frac),
            })

        info["entities"] = entities_meta

        # players meta
        players_meta = []
        for p in self.battle.players:
            players_meta.append({
                "player_id": p.player_id,
                "elixir": float(p.elixir),
                "elixir_waste": float(getattr(p, 'elixir_wasted', 0.0)),
                "hand": list(p.hand),
                "crowns": int(p.get_crown_count()),
                "king_hp": float(p.king_tower_hp),
                "left_hp": float(p.left_tower_hp),
                "right_hp": float(p.right_tower_hp),
            })
        info["players"] = players_meta
        # aggregate elixir waste (both players)
        info["elixir_waste"] = sum([getattr(p, 'elixir_wasted', 0.0) for p in self.battle.players])

        # players meta
        players_meta = []

        for p in self.battle.players:
            players_meta.append({
                "player_id": p.player_id,
                "elixir": float(p.elixir),
                "elixir_waste": float(getattr(p, 'elixir_wasted', 0.0)),
                "hand": list(p.hand),
                "crowns": int(p.get_crown_count()),
                "king_hp": float(p.king_tower_hp),
                "left_hp": float(p.left_tower_hp),
                "right_hp": float(p.right_tower_hp),
            })
        info["players"] = players_meta
        info["elixir_waste"] = sum([getattr(p, 'elixir_wasted', 0.0) for p in self.battle.players])

        return obs, info
    
    def decode_and_deploy(self, player_id: int, action: Optional[int] = None):
        # Special sentinel: -1 means "skip deploy" for this player (no play this tick)
        if action == self.no_op_action or action is None:
            # Return a standardized no-op result:
            #this is a successful no-op, not an invalid action, so success=True but other fields are -1 or None
            return -1, None, -1, -1, -1.0, -1.0, True

        card_idx = action // self.actions_per_tile
        tile_index = action % self.actions_per_tile
        x_tile = tile_index % self.tiles_x
        y_tile = tile_index // self.tiles_x

        # canonicalize Y coordinate so that player 0 always sees the bottom as 0
        if player_id == 1:
            y_tile = self.tiles_y - 1 - y_tile

        # integer cast
        card_idx = int(card_idx)
        x_tile = int(x_tile)
        y_tile = int(y_tile)

        # safe card selection: if the decoded card index is not present in the player's hand,
        # treat the action as a no-op rather than remapping to card 0.
        hand = self.battle.players[player_id].hand
        if card_idx < 0 or card_idx >= len(hand):
            # invalid action -> standardized no-op
            return -1, None, -1, -1, -1.0, -1.0, False
        card_name = hand[card_idx]

        # deploy position (center of tile)
        deploy_x = float(x_tile + 0.5)
        deploy_y = float(y_tile + 0.5)

        action_success = self.engine.simulate_action(player_id, card_name, deploy_x, deploy_y)

        return card_idx, card_name, x_tile, y_tile, deploy_x, deploy_y, action_success

    
    def step(self, action: int):
        
        # Decode and deploy for both players
        action0 = action

        p0_card_idx, card_name, p0_x_tile, p0_y_tile, deploy_x, deploy_y, p0_action_success = self.decode_and_deploy(0, action0)
        
        #opponents action is flipped on the y axis
        if self.opponent_policy is not None:
            # Use the opponent policy model to select an action
            opponent_obs = self.transpose_observation(self._render_obs())
            processed_opp_obs = self.opponent_policy.preprocess_observation(opponent_obs)

            raw_action1 = self.opponent_policy.predict(processed_opp_obs)
            action1 = self.opponent_policy.postprocess_action(raw_action1)
        else:
            action1 = self.action_space.sample()
        p1_card_idx, p1_card_name, p1_x_tile, p1_y_tile, p1_deploy_x, p1_deploy_y, p1_action_success = self.decode_and_deploy(1, action1)
        # Advance simulation one tick
        if not p1_action_success:
            print(f"Opponent action failed to deploy: {action1} -> card_idx {p1_card_idx}, tile ({p1_x_tile}, {p1_y_tile})")
        self.battle.step(self.speed_factor)
        self._step_count += 1

        # Compute reward: reward is always for player_0
        observation = self._render_obs()
        #NOTE: COMMENTED BECAUSE THIS IS HANDLED IN THE INFERENCE SCRIPT - THIS DESIGN MAY CHANGE
        # observations["player_1"] = self.transpose_obs(observations["player_0"])
        score = self._compute_score()
        reward = score - self._prev_score
        self._prev_score = score

        terminated = self.battle.game_over
        truncated = self._step_count >= self._max_steps

        info: Dict[str, Any] = {
            "tick": self.battle.tick,
            "time": self.battle.time,
            "last_action": {
            "player_0": {
                "card_idx": int(p0_card_idx),
                "card_name": str(card_name),
                "tile": (int(p0_x_tile), int(p0_y_tile)),
                "position": (deploy_x, deploy_y),
                "success": bool(p0_action_success),
            },
            "player_1": {
                "card_idx": int(p1_card_idx) if len(self.battle.players[1].hand) > 0 else None,
                "card_name": str(p1_card_name) if len(self.battle.players[1].hand) > 0 else None,
                "tile": (int(p1_x_tile), int(p1_y_tile)) if len(self.battle.players[1].hand) > 0 else None,
                "position": (p1_deploy_x, p1_deploy_y) if len(self.battle.players[1].hand) > 0 else None,
                "success": bool(p1_action_success) if len(self.battle.players[1].hand) > 0 else None,
            }
            }
        }

        # Provide structured entity metadata (stable for debugging / agents)
        entities_meta = []
        arena = self.battle.arena
        for eid, ent in self.battle.entities.items():
            card_stats = getattr(ent, 'card_stats', None)
            type_name = getattr(card_stats, 'name', None) if card_stats is not None else ent.__class__.__name__
            type_id = self._type_to_id.get(type_name, 0)

            px = int((ent.position.x / arena.width) * self.obs_shape[1])
            py = int((ent.position.y / arena.height) * self.obs_shape[0])
            px = max(0, min(self.obs_shape[1] - 1, px))
            py = max(0, min(self.obs_shape[0] - 1, py))

            try:
                hp_frac = max(0.0, min(1.0, ent.hitpoints / max(1.0, getattr(ent, 'max_hitpoints', 1.0))))
            except Exception:
                hp_frac = 1.0

            entities_meta.append({
                "id": getattr(ent, 'id', None),
                "type_name": type_name,
                "type_id": int(type_id),
                "player_id": getattr(ent, 'player_id', None),
                "position": (float(ent.position.x), float(ent.position.y)),
                "pixel": (int(px), int(py)),
                "hp": float(getattr(ent, 'hitpoints', 0)),
                "hp_frac": float(hp_frac),
            })

        info["entities"] = entities_meta

        # players meta
        players_meta = []
        for p in self.battle.players:
            players_meta.append({
                "player_id": p.player_id,
                "elixir": float(p.elixir),
                "elixir_waste": float(getattr(p, 'elixir_wasted', 0.0)),
                "hand": list(p.hand),
                "crowns": int(p.get_crown_count()),
                "king_hp": float(p.king_tower_hp),
                "left_hp": float(p.left_tower_hp),
                "right_hp": float(p.right_tower_hp),
            })

        info["players"] = players_meta
        info["elixir_waste"] = sum([getattr(p, 'elixir_wasted', 0.0) for p in self.battle.players])

        #reshape for pettingzoo
        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array"):
        #NOTE: DNU
        if mode == "rgb_array":
            return self._render_obs()
        return None

    def close(self):
        # Nothing special required for now
        return

    def _compute_score(self):
        """
        Compute a simple score based on tower HP differences and crowns.
        Positive score favors player 0, negative favors player 1."""
        p0 = self.battle.players[0]
        p1 = self.battle.players[1]

        tower_diff = (
            p0.king_tower_hp + p0.left_tower_hp + p0.right_tower_hp
            - p1.king_tower_hp - p1.left_tower_hp - p1.right_tower_hp
        )

        tower_term = tower_diff / MAX_TOTAL_TOWER_HP
        crown_term = (p0.get_crown_count() - p1.get_crown_count()) * 1.0

        return tower_term + crown_term


    def _render_obs(self) -> np.ndarray:
        """Render a 128x128x3 observation tensor with channels:

        - channel 0: owner mask (255 for player0, 128 for player1, 0 background)
        - channel 1: unit type id (0 background, 1..254 mapped per unit name)
        - channel 2: HP fraction encoded 0..255

        This keeps visualization concerns separate from the observation encoding.
        """
        img_w, img_h = self.obs_shape[1], self.obs_shape[0]
        img = np.zeros(self.obs_shape, dtype=np.uint8)

        arena = self.battle.arena
        for entity in self.battle.entities.values():
            ex = entity.position.x
            ey = entity.position.y

            # Project into pixel coordinates
            px = int((ex / arena.width) * img_w)
            py = int((ey / arena.height) * img_h)

            # Clamp
            px = max(0, min(img_w - 1, px))
            py = max(0, min(img_h - 1, py))

            # Small square size in pixels
            size_x = max(1, int(img_w / arena.width / 2))
            size_y = max(1, int(img_h / arena.height / 2))

            x0 = max(0, px - size_x)
            x1 = min(img_w - 1, px + size_x)
            y0 = max(0, py - size_y)
            y1 = min(img_h - 1, py + size_y)

            # Owner channel: 255 for p0, 128 for p1
            owner_val = 255 if getattr(entity, 'player_id', 0) == 0 else 128
            img[y0:y1, x0:x1, 0] = owner_val

            # Unit type id mapping
            card_stats = getattr(entity, 'card_stats', None)
            if card_stats is not None:
                type_name = getattr(card_stats, 'name', None) or entity.__class__.__name__
            else:
                type_name = entity.__class__.__name__

            type_id = self._type_to_id.get(type_name)
            if type_id is None:
                # Assign new id (cap at 254)
                if self._next_type_id < 255:
                    type_id = self._next_type_id
                    self._type_to_id[type_name] = type_id
                    self._next_type_id += 1
                else:
                    type_id = 254

            img[y0:y1, x0:x1, 1] = int(type_id)

            # HP channel (normalized)
            try:
                hp_frac = max(0.0, min(1.0, entity.hitpoints / max(1.0, getattr(entity, 'max_hitpoints', 1.0))))
            except Exception:
                hp_frac = 1.0
            img[y0:y1, x0:x1, 2] = int(hp_frac * 255)

        return img

class ClashRoyaleVectorEnv(AsyncVectorEnv):
    """Vectorized wrapper around multiple `ClashRoyaleGymEnv` instances.

    This wrapper instantiates `num_envs` independent `ClashRoyaleGymEnv`
    environments and exposes a simple batched `reset` / `step` API compatible
    with `gymnasium.vector.VectorEnv`.

    Notes:
    - Actions must be a 1-D array-like of ints with shape `(num_envs,)` where
        each entry is the discrete action for the corresponding environment's
        `player_0` agent. The opponent action is left as `None` (the wrapped
        env will either sample or use an internal opponent policy).
    - Observations are returned as an array with shape
        `(num_envs, H, W, C)` where `H,W,C` match the wrapped env's `obs_shape`.
    """

    def __init__(self,
                    num_envs: int,
                    opponent_policies: Optional[list[InferenceModel]] = None,
                    **env_kwargs):
        # Total number of environments
        self.num_envs = num_envs
        self.env_fns = []
       
        for i in range(self.num_envs):
            def make_env(op_policy):
                def _init():
                    env = ClashRoyaleGymEnv(**env_kwargs)
                    if op_policy:
                        env.set_opponent_policy(op_policy)
                    return env
                return _init
            opponent_policy = None
            if opponent_policies:
                opponent_policy = opponent_policies[i % len(opponent_policies)]

            self.env_fns.append(make_env(opponent_policy))
            

        super().__init__(self.env_fns)

