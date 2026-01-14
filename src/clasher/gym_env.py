"""Gymnasium-compatible environment wrapping the Clash Royale battle engine.

This provides a minimal, working `ClashRoyaleGymEnv` implementation suitable
for running the project's `helloworld.py` demo and for basic RL loops.
"""
from typing import Tuple, Dict, Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError("gymnasium is required to use ClashRoyaleGymEnv.\nInstall it with `pip install gymnasium`.")

from .engine import BattleEngine
from .arena import TileGrid, Position


class ClashRoyaleGymEnv(gym.Env):
    """Simple Gymnasium environment wrapper around BattleEngine/BattleState.

    Action encoding (discrete 2304): card_idx (4) x x_tile (18) x y_tile (32)
    Observation: dict with `'p1-view'` -> 128x128x3 uint8 RGB-like tensor
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, speed_factor: float = 1.0, data_file: str = "gamedata.json", max_steps: int = 9090):
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
        self.action_space = spaces.Discrete(self.num_cards * self.actions_per_tile)

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

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            return None
        np.random.seed(seed)
        return seed

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
                "hand": list(p.hand),
                "crowns": int(p.get_crown_count()),
                "king_hp": float(p.king_tower_hp),
                "left_hp": float(p.left_tower_hp),
                "right_hp": float(p.right_tower_hp),
            })
        info["players"] = players_meta

        # players meta
        players_meta = []
        for p in self.battle.players:
            players_meta.append({
                "player_id": p.player_id,
                "elixir": float(p.elixir),
                "hand": list(p.hand),
                "crowns": int(p.get_crown_count()),
                "king_hp": float(p.king_tower_hp),
                "left_hp": float(p.left_tower_hp),
                "right_hp": float(p.right_tower_hp),
            })
        info["players"] = players_meta

        return {"p1-view": obs}, info

    def step(self, action: int):
        # Decode action
        card_idx = action // self.actions_per_tile
        tile_index = action % self.actions_per_tile
        x_tile = tile_index % self.tiles_x
        y_tile = tile_index // self.tiles_x

        # Choose card name from player hand (fall back safely)
        hand = self.battle.players[0].hand
        if card_idx < 0 or card_idx >= len(hand):
            card_idx = 0
        card_name = hand[card_idx]

        # Convert to deploy position (center of tile)
        deploy_x = float(x_tile + 0.5)
        deploy_y = float(y_tile + 0.5)

        # Attempt to deploy as player 0
        self.engine.simulate_action(0, card_name, deploy_x, deploy_y)

        # Advance simulation one tick
        self.battle.step(self.speed_factor)
        self._step_count += 1

        # Compute observation, reward, termination
        obs = self._render_obs()
        score = self._compute_score()
        reward = score - self._prev_score
        self._prev_score = score

        terminated = bool(self.battle.game_over)
        truncated = self._step_count >= self._max_steps

        info: Dict[str, Any] = {
            "tick": self.battle.tick,
            "time": self.battle.time,

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
                "hand": list(p.hand),
                "crowns": int(p.get_crown_count()),
                "king_hp": float(p.king_tower_hp),
                "left_hp": float(p.left_tower_hp),
                "right_hp": float(p.right_tower_hp),
            })

        info["players"] = players_meta

        return {"p1-view": obs}, float(reward), terminated, truncated, info

    def render(self, mode: str = "rgb_array"):
        if mode == "rgb_array":
            return self._render_obs()
        return None

    def close(self):
        # Nothing special required for now
        return

    def _compute_score(self) -> float:
        """Compute a simple scalar score combining crowns and tower HP differences."""
        p0 = self.battle.players[0]
        p1 = self.battle.players[1]

        crowns_diff = p0.get_crown_count() - p1.get_crown_count()
        tower_hp_diff = (p0.king_tower_hp + p0.left_tower_hp + p0.right_tower_hp) - (
            p1.king_tower_hp + p1.left_tower_hp + p1.right_tower_hp
        )

        # Scale components so rewards are in reasonable ranges
        return float(crowns_diff * 100.0 + tower_hp_diff / 100.0)

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
