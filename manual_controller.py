#!/usr/bin/env python3
"""Manual controller using Pygame to play as player 0.

Usage:
  python3 manual_controller.py [--opponent none|random|recurrent] [--model PATH]

Click a card in the right UI panel to select it, then click a tile in the arena
to deploy. The script routes actions through `ClashRoyaleGymEnv.step(...)` so an
opponent policy (random or a loaded recurrent PPO) can act for player 1.
"""
import argparse
import pygame
import sys
from typing import Optional
import os
import json
import datetime

from visualize_battle import BattleVisualizer, ARENA_X, ARENA_Y, TILE_SIZE
from src.clasher.arena import Position
from src.clasher.gym_env import ClashRoyaleGymEnv

TIME_REDUCE_FACTOR = 1 #real time to simulation time
try:
    from wrappers.randompolicy import RandomPolicyInferenceModel
except Exception:
    RandomPolicyInferenceModel = None

try:
    from wrappers.recurrentppo import RecurrentPPOInferenceModel
except Exception:
    RecurrentPPOInferenceModel = None

DECK = ["Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons", "Log"]

class ManualController(BattleVisualizer):
    def __init__(self, opponent_type: str = "none", model_path: Optional[str] = None, turn_based: bool = False, step_count: int = 1):
        super().__init__()

        # Use BattleVisualizer's engine/battle created in super().__init__
        # (BattleVisualizer already set up self.engine and self.battle)
        # Pygame state
        self.selected_card_idx: Optional[int] = None
        self.hover_tile = None
        self.paused = False
        # Turn-based mode
        self.turn_based = turn_based
        self.step_requested = False
        self.step_count = int(step_count or 1)
        # On-screen step-count input state
        self.step_input_active = False
        self.step_input_text = str(self.step_count)

        # Create a Gym env wrapper to leverage opponent policy handling
        self.env = ClashRoyaleGymEnv()
        # Use the visualizer's engine/battle so env.step manipulates the same state
        # (ClashRoyaleGymEnv.decode_and_deploy uses self.engine.simulate_action)
        self.env.engine = self.engine
        self.env.engine.battle = self.battle
        self.env.battle = self.battle

        #TODO - HARDCODING DECKS FOR BOTH PLAYERS
        self.env.set_player_deck(0, DECK)
        self.env.set_player_deck(1, DECK)

        # Load opponent policy
        self.opponent = None
        if opponent_type == "random" and RandomPolicyInferenceModel is not None:
            self.opponent = RandomPolicyInferenceModel(self.env, player_id=1)
            self.env.set_opponent_policy(self.opponent)
        elif opponent_type == "recurrent" and RecurrentPPOInferenceModel is not None and model_path:
            # RecurrentPPOInferenceModel expects a model path in its constructor
            try:
                self.opponent = RecurrentPPOInferenceModel(model_path, eval=True, deterministic=True)
                self.env.set_opponent_policy(self.opponent)
            except Exception as e:
                print(f"Failed loading recurrent model: {e}")
        elif opponent_type == "recurrent_onnx":
            from wrappers.rppo_onnx import RecurrentPPOONNXInferenceModel
            try:
                self.opponent = RecurrentPPOONNXInferenceModel(model_path)
                self.env.set_opponent_policy(self.opponent)
            except Exception as e:
                print(f"Failed loading ONNX model: {e}")

        # UI layout for hand cards
        self.ui_card_rects = []
        self._build_card_ui()
        # Step button and last action display
        self.step_button_rect = None
        self.last_action_info = None

        # Setup logging
        os.makedirs("replays/logs", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"replays/logs/manual_{ts}.jsonl"
        try:
            self.log_fh = open(self.log_path, "a")
        except Exception as e:
            print(f"Failed to open log file {self.log_path}: {e}")
            self.log_fh = None
        self.last_logged_tick = -1


    def _build_card_ui(self):
        # 4 card rects in the right panel (placed near bottom to avoid overlapping header)
        panel_x = ARENA_X + 18 * TILE_SIZE + 30
        w = 140
        h = 60
        gap = 10
        # Compute bottom-aligned Y inside right UI panel (panel drawn in draw_ui uses ARENA_Y + ARENA_HEIGHT)
        panel_bottom = ARENA_Y + 32 * TILE_SIZE
        total_h = 4 * h + 3 * gap
        panel_y = max(ARENA_Y + 60, panel_bottom - total_h - 30)
        self.ui_card_rects = []
        for i in range(4):
            r = pygame.Rect(panel_x, panel_y + i * (h + gap), w, h)
            self.ui_card_rects.append(r)
        # Step button above the cards
        btn_h = 36
        btn_w = 100
        btn_x = panel_x
        btn_y = panel_y - btn_h - 10
        self.step_button_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        # Step-count input box to the right of the cards
        input_w = 96
        input_h = 36
        input_x = panel_x + w + 12
        # place roughly centered vertically relative to card stack
        input_y = panel_y + (h * 2) - (input_h // 2)
        self.step_input_rect = pygame.Rect(input_x, input_y, input_w, input_h)

    def world_pos_to_tile(self, mx: int, my: int):
        # Convert mouse pixel to tile coordinates (x in 0..17, y in 0..31)
        rel_x = mx - ARENA_X
        rel_y = my - ARENA_Y
        if rel_x < 0 or rel_y < 0:
            return None
        tx = int(rel_x // TILE_SIZE)
        ty = int(rel_y // TILE_SIZE)
        if tx < 0 or tx >= 18 or ty < 0 or ty >= 32:
            return None
        return tx, ty

    def draw_ui(self):
        # Use parent UI and then draw hand cards interactively
        super().draw_ui()
        # Draw turn-based hint if enabled
        if getattr(self, 'turn_based', False):
            hint = self.small_font.render("TURN MODE: press N to step", True, (0, 0, 0))
            self.screen.blit(hint, (ARENA_X + 18 * TILE_SIZE + 40, ARENA_Y + 20))
        import pygame
        # Draw Step button
        if self.step_button_rect:
            pygame.draw.rect(self.screen, (180, 180, 180), self.step_button_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), self.step_button_rect, 2)
            label = self.small_font.render("STEP", True, (0, 0, 0))
            lab_rect = label.get_rect(center=self.step_button_rect.center)
            self.screen.blit(label, lab_rect)

        # Draw step-count input box (editable)
        if getattr(self, 'step_input_rect', None):
            # background
            pygame.draw.rect(self.screen, (245, 245, 245), self.step_input_rect)
            # border - active highlight
            border_col = (0, 120, 215) if getattr(self, 'step_input_active', False) else (0, 0, 0)
            pygame.draw.rect(self.screen, border_col, self.step_input_rect, 2)
            # label
            label = self.small_font.render("Steps:", True, (0, 0, 0))
            self.screen.blit(label, (self.step_input_rect.x - 58, self.step_input_rect.y + 6))
            # value text
            display_text = getattr(self, 'step_input_text', str(self.step_count))
            txt_surf = self.small_font.render(display_text, True, (0, 0, 0))
            txt_rect = txt_surf.get_rect()
            txt_pos = (self.step_input_rect.x + 8, self.step_input_rect.y + (self.step_input_rect.height - txt_rect.height) // 2)
            self.screen.blit(txt_surf, txt_pos)

        # Draw last opponent action
        if self.last_action_info:
            # Display player_1 last action
            la = self.last_action_info.get('player_1') if isinstance(self.last_action_info, dict) else None
            if la:
                text = f"P1: {la.get('card_name')} @ {la.get('tile')} {'OK' if la.get('success') else 'FAIL'}"
            else:
                text = "P1: -"
            action_text = self.small_font.render(text, True, (0, 0, 0))
            self.screen.blit(action_text, (ARENA_X + 18 * TILE_SIZE + 40, ARENA_Y + 60))
        # Draw card slots and names
        for idx, rect in enumerate(self.ui_card_rects):
            color = (200, 200, 200)
            border = (0, 0, 0)
            if self.selected_card_idx == idx:
                border = (0, 200, 0)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, border, rect, 3)

            # Draw card name from player's hand
            hand = self.battle.players[0].hand
            name = hand[idx] if idx < len(hand) else "-"
            text = self.small_font.render(name, True, (0, 0, 0))
            tx = rect.x + 8
            ty = rect.y + 8
            self.screen.blit(text, (tx, ty))

            # Mana cost (try to read stats)
            try:
                cs = self.battle.card_loader.get_card(name)
                cost = str(int(cs.mana_cost)) if cs else "?"
            except Exception:
                cost = "?"
            cost_text = self.small_font.render(f"Cost: {cost}", True, (0, 0, 0))
            self.screen.blit(cost_text, (rect.x + 8, rect.y + 32))

        # small hint for editing input when active
        if getattr(self, 'step_input_active', False):
            hint = self.small_font.render("Type digits, Enter to apply", True, (80, 80, 80))
            hx = self.step_input_rect.x
            hy = self.step_input_rect.y + self.step_input_rect.height + 6
            self.screen.blit(hint, (hx, hy))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                # If step-count input is active, route keys there first
                if getattr(self, 'step_input_active', False):
                    # Allow digits, backspace, enter, and escape to cancel
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        # Commit value
                        try:
                            v = int(self.step_input_text)
                            self.step_count = max(1, v)
                        except Exception:
                            self.step_count = max(1, int(self.step_count))
                        self.step_input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.step_input_text = self.step_input_text[:-1]
                        if self.step_input_text == "":
                            self.step_input_text = ""
                    elif event.key == pygame.K_ESCAPE:
                        # cancel editing revert to current count
                        self.step_input_text = str(self.step_count)
                        self.step_input_active = False
                    else:
                        # Accept numeric keys
                        if pygame.K_0 <= event.key <= pygame.K_9:
                            digit = event.key - pygame.K_0
                            # Append digit
                            if not (len(self.step_input_text) == 1 and self.step_input_text == "0"):
                                self.step_input_text += str(digit)
                            else:
                                self.step_input_text = str(digit)
                    # consume key
                    continue
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_n and self.turn_based:
                    self.step_requested = True
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    # increase step count
                    self.step_count = min(10000, self.step_count + 1)
                elif event.key == pygame.K_MINUS:
                    self.step_count = max(1, self.step_count - 1)
                elif pygame.K_0 <= event.key <= pygame.K_9:
                    # set step count directly from digit keys (0 sets to 10)
                    digit = event.key - pygame.K_0
                    self.step_count = 10 if digit == 0 else digit
                elif event.key == pygame.K_r:
                    self.engine = self.engine
                    self.battle = self.engine.create_battle()
                    # keep env in sync with the engine/battle used by visualizer
                    self.env.engine = self.engine
                    self.env.engine.battle = self.battle
                    self.env.battle = self.battle
                    self._build_card_ui()
                else:
                    # allow parent hotkeys
                    super().handle_events()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # Focus/unfocus step-count input
                if getattr(self, 'step_input_rect', None) and self.step_input_rect.collidepoint(mx, my):
                    self.step_input_active = True
                    # place caret at end
                    self.step_input_text = str(self.step_input_text or '')
                    continue
                else:
                    # clicking outside input deactivates it
                    if getattr(self, 'step_input_active', False):
                        self.step_input_active = False
                        # revert empty to current count
                        if not self.step_input_text:
                            self.step_input_text = str(self.step_count)
                # Check if clicked a UI card
                for idx, rect in enumerate(self.ui_card_rects):
                    if rect.collidepoint(mx, my):
                        # Toggle selection
                        if self.selected_card_idx == idx:
                            self.selected_card_idx = None
                        else:
                            self.selected_card_idx = idx
                        break
                # Step button click
                if self.step_button_rect and self.step_button_rect.collidepoint(mx, my):
                    self.step_requested = True
                    break
                else:
                    # Click on arena: try to deploy if card selected
                    tile = self.world_pos_to_tile(mx, my)
                    if tile and self.selected_card_idx is not None:
                        tx, ty = tile
                        # Canonicalize y such that player 0 sees bottom as 0 (same as env.decode_and_deploy)
                        # Our tiles' origin matches env expectations already
                        card_idx = int(self.selected_card_idx)
                        tiles_x = self.env.tiles_x
                        tile_index = ty * tiles_x + tx
                        action = card_idx * (self.env.tiles_x * self.env.tiles_y) + tile_index
                        # print(action)
                        # Step env with manual action - env will also run opponent policy
                        try:
                            obs, reward, terminated, truncated, info = self.env.step(int(action))
                            # Log env-provided info for this action
                            self._log_info_from_env(info)
                        except Exception as e:
                            print(f"Error stepping env: {e}")
                        # clear selection after attempt
                        self.selected_card_idx = None
                        # If in turn-based mode, clear automatic stepping request
                        if self.turn_based:
                            self.step_requested = False

        return True

    def run(self):
        import time
        print("Manual controller started. Click a card then a tile to play.")
        running = True
        last_time = time.time()
        battle_accumulator = 0.0
        battle_timestep = 1.0 / 30.0 / TIME_REDUCE_FACTOR  # match env default timestep (30 FPS) scaled by time factor

        while running:
            running = self.handle_events()

            current_time = time.time()
            frame_time = current_time - last_time
            last_time = current_time

            if not self.paused and not self.battle.game_over:
                battle_accumulator += frame_time
                while battle_accumulator >= battle_timestep:
                    # Advance simulation via env.step with a SKIP sentinel so opponent policy runs
                    if self.turn_based:
                        # Only step when requested; perform `step_count` steps
                        if self.step_requested:
                            steps = max(1, int(self.step_count))
                            for _ in range(steps):
                                try:
                                    obs, reward, terminated, truncated, info = self.env.step(-1)
                                    if terminated:
                                        break
                                    self._log_info_from_env(info)
                                except Exception:
                                    self.battle.step(self.speed if hasattr(self, 'speed') else 1.0)
                                    self._log_tick_state()
                            self.step_requested = False
                        else:
                            # In turn-based mode do not advance time
                            pass
                    else:
                        try:
                            obs, reward, terminated, truncated, info = self.env.step(-1)
                            # Log env info (includes last_action for both players)
                            self._log_info_from_env(info)
                        except Exception:
                            # Fallback to direct step if env stepping fails
                            self.battle.step(self.speed if hasattr(self, 'speed') else 1.0)
                            self._log_tick_state()
                    battle_accumulator -= battle_timestep

            # Draw
            self.screen.fill((255, 255, 255))
            self.draw_arena()
            self.draw_towers()
            self.draw_entities()
            self.draw_ui()

            # Highlight hovered tile
            mx, my = pygame.mouse.get_pos()
            tile = self.world_pos_to_tile(mx, my)
            if tile:
                tx, ty = tile
                rect = pygame.Rect(ARENA_X + tx * TILE_SIZE, ARENA_Y + ty * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                s = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                s.fill((255, 255, 0, 80))
                self.screen.blit(s, rect)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        # Close log file
        if getattr(self, 'log_fh', None):
            self.log_fh.close()

    def _log_info_from_env(self, info: dict):
        """Write an info dict returned from `env.step` to the JSONL log."""
        if not getattr(self, 'log_fh', None) or not isinstance(info, dict):
            return
        tick = info.get('tick', self.battle.tick)
        if tick == self.last_logged_tick:
            return
        entry = {
            'tick': tick,
            'time': info.get('time', self.battle.time),
            'last_action': info.get('last_action'),
            'players': info.get('players', []),
            'entities': info.get('entities', [])
        }
        try:
            self.log_fh.write(json.dumps(entry) + "\n")
            self.log_fh.flush()
            self.last_logged_tick = tick
            # store last action for UI display
            try:
                self.last_action_info = entry.get('last_action')
            except Exception:
                self.last_action_info = None
        except Exception as e:
            print(f"Failed writing log entry: {e}")

    def _log_tick_state(self):
        """Log current battle tick/state when stepping directly via `BattleState.step`."""
        if not getattr(self, 'log_fh', None):
            return
        tick = getattr(self.battle, 'tick', None)
        if tick is None or tick == self.last_logged_tick:
            return
        # Build players meta
        players_meta = []
        for p in self.battle.players:
            players_meta.append({
                'player_id': p.player_id,
                'elixir': float(p.elixir),
                'hand': list(p.hand),
                'crowns': int(p.get_crown_count()),
                'king_hp': float(p.king_tower_hp),
                'left_hp': float(p.left_tower_hp),
                'right_hp': float(p.right_tower_hp)
            })

        entry = {
            'tick': tick,
            'time': float(getattr(self.battle, 'time', 0.0)),
            'last_action': None,
            'players': players_meta,
            'entities': len(self.battle.entities)
        }
        try:
            self.log_fh.write(json.dumps(entry) + "\n")
            self.log_fh.flush()
            self.last_logged_tick = tick
        except Exception as e:
            print(f"Failed writing tick log: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--opponent", choices=["none", "random", "recurrent", "recurrent_onnx"], default="none")
    p.add_argument("--model", type=str, default=None, help="Path to opponent model (for recurrent)")
    p.add_argument("--turn_based", action="store_true", help="Run in turn-based mode; press N to advance one tick")
    p.add_argument("--step-count", type=int, default=1, help="Default number of ticks to advance per step (N or STEP)")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ctrl = ManualController(opponent_type=args.opponent, model_path=args.model, turn_based=args.turn_based, step_count=args.step_count)
    ctrl.run()
