#!/usr/bin/env python3
"""Play back an action log and visualize using the existing BattleVisualizer.

Usage: python3 scripts/play_replay.py replays/logs/actions_YYYYMMDD_HHMMSS.jsonl
"""
import json
import sys
import time
import os
from typing import List, Dict

from visualize_battle import BattleVisualizer
from src.clasher.arena import Position


def load_actions(path: str) -> List[Dict]:
    actions = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            actions.append(json.loads(line))
    return actions


def main(path: str):
    actions = load_actions(path)
    if not actions:
        print("No actions found in log")
        return

    vis = BattleVisualizer()

    # We'll run the visualizer display loop but drive the battle by applying
    # recorded actions at the ticks they were logged.
    # Build a mapping tick -> list of actions
    schedule = {}
    for entry in actions:
        t = entry.get('tick')
        if t is None:
            continue
        schedule.setdefault(int(t), []).append(entry.get('last_action'))

    print(f"Loaded {len(actions)} logged steps. Scheduled actions for {len(schedule)} ticks.")

    # Use the visualizer's existing window and draw loop but step manually
    import pygame
    pygame.init()
    running = True
    clock = pygame.time.Clock()

    tick = 0
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # Apply scheduled actions for this tick
        scheduled = schedule.get(tick, [])
        for act in scheduled:
            if not act:
                continue
            # act contains card_name, position and success flag
            card = act.get('card_name')
            pos = act.get('position')
            if card and pos:
                # Deploy on visualizer's battle instance
                vis.battle.deploy_card(0, card, Position(pos[0], pos[1]))

        # Step the battle once
        vis.battle.step(speed_factor=1.0)

        # Draw
        vis.screen.fill((255,255,255))
        vis.draw_arena()
        vis.draw_towers()
        vis.draw_entities()
        vis.draw_ui()
        pygame.display.flip()

        tick += 1
        clock.tick(30)

        # Stop when no more scheduled ticks and battle over
        if tick > max(schedule.keys()) + 200 or vis.battle.game_over:
            time.sleep(1.0)
            running = False

    pygame.quit()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: play_replay.py <log.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
