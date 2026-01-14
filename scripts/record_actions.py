#!/usr/bin/env python3
"""Run the Gym env for a number of steps and log attempted actions to a JSONL file.

Usage: python3 scripts/record_actions.py [steps]
"""
import json
import time
import os
import sys
from datetime import datetime

from src.clasher.gym_env import ClashRoyaleGymEnv


def main(steps: int = 200):
    env = ClashRoyaleGymEnv()
    obs, info = env.reset()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"replays/logs"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"actions_{timestamp}.jsonl")

    print(f"Logging actions to {out_file}")

    with open(out_file, 'w') as fh:
        for step in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # info['last_action'] added by env
            entry = {
                "tick": info.get("tick", None),
                "time": info.get("time", None),
                "last_action": info.get("last_action", None),
                "players": info.get("players", None),
            }
            fh.write(json.dumps(entry) + "\n")

            if terminated or truncated:
                print("Episode finished")
                break

    print("Done")


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(steps)
