"""
Pass in a model + env wrapper. Wrapper will handle predictions via model.predict().
"""

import os
import json
import datetime
import logging
import matplotlib.pyplot as plt

from src.clasher.gym_env import ClashRoyaleGymEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from inference.wrappers.ppo import PPOInferenceModel
from scripts.train.ppo_wrapper import PPOObsWrapper  # TODO: remove hardcoding

# --------------------------
# Setup logging
# --------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --------------------------
# Environment
# --------------------------
base_env = ClashRoyaleGymEnv()
# Wrap for PPO
wrapped_env = DummyVecEnv([lambda: PPOObsWrapper(base_env)])

# --------------------------
# Prepare logging to JSONL
# --------------------------
os.makedirs("replays/logs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"replays/logs/inference_{timestamp}.jsonl"
log_fh = open(log_path, "a")
logging.info(f"Logging actions to {log_path}")

# --------------------------
# Load PPO model
# --------------------------
ppo_model = PPOInferenceModel(wrapped_env)  # pass base env
ppo_model.load_model("models/ppo_clashroyale_baseline.zip")

# --------------------------
# Run inference
# --------------------------
obs = ppo_model.reset()  # obs = wrapped, info = base env info
num_steps = 0

while True:
    # Predict action
    action = ppo_model.model.predict(obs)[0]  # deterministic=True by default

    # Step through env
    obs, reward, terminated, info = ppo_model.env.step(action)
    #get the info
    base_info = info[0]
    # Log step
    entry = {
        "tick": base_info.get("tick"),
        "time": base_info.get("time"),
        "last_action": base_info.get("last_action"),
        "players": base_info.get("players"),
        "reward": float(reward),
    }
    log_fh.write(json.dumps(entry) + "\n")
    log_fh.flush()

    if terminated:
        logging.info("Game terminated. Exiting the loop.")
        break

    # Example: log player and entity info every 1000 steps
    if num_steps % 1000 == 0:
        logging.info(f"Step: {num_steps}, Reward: {reward}")

        # Towers info
        for eid, ent in base_env.battle.entities.items():
            name = getattr(getattr(ent, "card_stats", None), "name", ent.__class__.__name__)
            if "tower" in name.lower():
                px = int((ent.position.x / base_env.battle.arena.width) * base_env.obs_shape[1])
                py = int((ent.position.y / base_env.battle.arena.height) * base_env.obs_shape[0])
                px = max(0, min(base_env.obs_shape[1] - 1, px))
                py = max(0, min(base_env.obs_shape[0] - 1, py))
                logging.info(f"Entity {eid} {name} pos ({ent.position.x:.1f},{ent.position.y:.1f}) pixel ({px},{py})")

        # Players info
        for player in base_info["players"]:
            logging.info(f"Player {player['player_id']} - Elixir: {player['elixir']}, Hand: {player['hand']}, Crowns: {player['crowns']}")

    num_steps += 1

# --------------------------
# Cleanup
# --------------------------
wrapped_env.close()
log_fh.close()
logging.info("Inference finished.")
