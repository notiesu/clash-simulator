"""
From base.py import sim_game method to run the environment loop and log actions to a JSONL file.
"""
from inference.wrappers.base import InferenceModel
from inference.wrappers.ppo import PPOInferenceModel
from stable_baselines3 import PPO
from src.clasher.gym_env import ClashRoyaleGymEnv
import logging
import argparse


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Clash Royale simulation with PPO Inference Model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PPO model file.")
    args = parser.parse_args()

    # Example usage
    env = ClashRoyaleGymEnv()
    #TODO - NOT ALWAYS PPO INFERENCE MODEL - MAKE SOME WAY TO SPECIFY
    model = PPOInferenceModel(env, printLogs=True)
    model.load_model(args.model_path)
    logging.info("Starting simulation game with PPO Inference Model.")
    model.sim_game()
    model.env.close()
    logging.info("Simulation game ended.")
