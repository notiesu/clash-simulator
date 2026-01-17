"""
From base.py import sim_game method to run the environment loop and log actions to a JSONL file.
"""
from inference.wrappers.base import InferenceModel
from inference.wrappers.ppo import PPOInferenceModel
from stable_baselines3 import PPO
from src.clasher.gym_env import ClashRoyaleGymEnv
import logging


if __name__ == "__main__":
    # Example usage
    env = ClashRoyaleGymEnv()
    model = PPOInferenceModel(env, printLogs=True)
    model.load_model("models/ppo_clashroyale_baseline.zip")
    logging.info("Starting simulation game with PPO Inference Model.")
    model.sim_game()
    model.env.close()
    logging.info("Simulation game ended.")
