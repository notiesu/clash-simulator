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
    schedule = []
    for entry in actions:
        t = entry.get('tick')
        if t is None:
            continue
        last_action = entry.get('last_action')
        if last_action:
            schedule.append(last_action)
        else:
            schedule.append({})

    # print(schedule[1])
    print(f"Loaded {len(actions)} logged steps. Scheduled actions for {len(schedule)} ticks.")
    print(schedule)
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
        scheduled = schedule[tick]
        # print(scheduled)
        if not scheduled:
            # print("Scheduled wrong format")
            continue
        for player, act in scheduled.items():
            if not act:
                continue
            # act contains card_idx, card_name, tile, position, and success flag
            card = act.get('card_name')
            pos = act.get('position')
            success = act.get('success')
            if card and pos and success:  # Only deploy if the action was successful
                # Deploy on visualizer's battle instance for the respective player
                # print(card + " " + str(pos) + " " + player)
                player_id = int(player.split('_')[1])  # Extract player ID from key
                vis.battle.deploy_card(player_id, card, Position(pos[0], pos[1]))
            else:
                # print(f"Skipping action for player {player}: incomplete data or unsuccessful deployment.")
                continue

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
        if tick > len(schedule) + 200 or vis.battle.game_over:
            time.sleep(1.0)
            running = False

    pygame.quit()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: play_replay.py <log.jsonl>")
        sys.exit(1)
    main(sys.argv[1])
