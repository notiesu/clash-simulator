"""
From base.py import sim_game method to run the environment loop and log actions to a JSONL file.
"""
from src.clasher.model import InferenceModel
from wrappers.ppo import PPOInferenceModel
from wrappers.recurrentppo import RecurrentPPOInferenceModel
from wrappers.randompolicy import RandomPolicyInferenceModel
from wrappers.rppo_onnx import RecurrentPPOONNXInferenceModel
from wrappers.behavior_cloning import BCArgs, BCInferenceModel
from stable_baselines3 import PPO
from src.clasher.gym_env import ClashRoyaleGymEnv
import logging
import argparse
import numpy as np

import os
import datetime
import json

"""
Method for transposing observation - switches p0 and p1 views
NOTE: Action transposition is handled within env step. See gym_env decode_and_deploy for implementation.
"""

DECK = ["Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons", "Log"]

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Clash Royale simulation with PPO Inference Model.")
    parser.add_argument("--p0_model_path", type=str, required=True, help="Path to the PPO model file.")
    parser.add_argument("--p0_model_type", type=str, default="PPO", help="Type of the model to use for inference (default: PPO).")
    parser.add_argument("--p1_model_path", type=str, required=True, help="Path to the model file.")
    parser.add_argument("--p1_model_type", type=str, default="PPO", help="Type of the model to use for inference (default: PPO).")
    parser.add_argument("--printLogs", action="store_true", help="Enable logging of actions to a JSONL file.")
    parser.add_argument("--p0_vocab_json", type=str, default=None, help="(BC only) Path to token2id json")
    parser.add_argument("--p1_vocab_json", type=str, default=None, help="(BC only) Path to token2id json")
    parser.add_argument("--history_len", type=int, default=20, help="(BC only) history length")
    parser.add_argument("--pad_id", type=int, default=0, help="(BC only) pad token id used during training")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    args = parser.parse_args()

    # Example usage
    env = ClashRoyaleGymEnv()

    #same for player 0 selection
    if args.p0_model_type == "PPO":
        model_p0 = PPOInferenceModel()
    elif args.p0_model_type == "RecurrentPPO":
        model_p0 = RecurrentPPOInferenceModel()
    elif args.p0_model_type == "RandomPolicy":
        model_p0 = RandomPolicyInferenceModel(env, player_id=0)
    elif args.p0_model_type == "BC":
        bc_args = BCArgs(
            token2id_path=args.p0_vocab_json,  # <-- updated name
            pad_id=args.pad_id,
            model_path=args.p0_model_path,
            history_len=args.history_len,
            device=args.device,
        )
        model_p0 = BCInferenceModel(env=env, bc_args=bc_args, printLogs=args.printLogs)
    else:
        raise ValueError(f"Unknown p0_model_type: {args.p0_model_type}")

    #same for player 1 selection
    if args.p1_model_type == "PPO":
        model_p1 = PPOInferenceModel()
    elif args.p1_model_type == "RecurrentPPO":
        model_p1 = RecurrentPPOInferenceModel()
    elif args.p1_model_type == "RandomPolicy":
        model_p1 = RandomPolicyInferenceModel(env, player_id=1)
    elif args.p1_model_type == "BC":
        bc_args = BCArgs(
            token2id_path=args.p1_vocab_json,  # <-- updated name
            pad_id=args.pad_id,
            history_len=args.history_len,
            device=args.device,
        )
        model_p1 = BCInferenceModel(env=env, bc_args=bc_args, printLogs=args.printLogs)
    else:
        raise ValueError(f"Unknown p1_model_type: {args.p1_model_type}")

    logging.info("Both models loaded successfully.")
    logging.info("Starting simulation game.")
    """
    MAIN ENVIRONMENT SIM LOOP
    """
    obs, info = env.reset()
    done = False
    num_steps = 0

    #set opponent policy for env
    env.set_opponent_policy(model_p1)
    env.set_player_deck(0, DECK)
    env.set_player_deck(1, DECK)

    if args.printLogs:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        logging.info("Environment reset. Starting the game loop.")

        # Prepare JSONL action log
        os.makedirs("replays/logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"replays/logs/inference_{timestamp}.jsonl"
        logging.info(f"Logging actions to {log_path}")
        log_fh = open(log_path, "a")


    while not done:
        # preprocess observations
        obs_p0 = model_p0.preprocess_observation(obs)

        #get actions
        action_p0 = model_p0.predict(obs_p0)

        #post process actions
        action_p0 = model_p0.postprocess_action(action_p0)

        obs, reward, done, truncated, info = env.step(action_p0)

        #env handles p1 action internally

        #post process reward 
        #NOTE: For now only processes reward for p0 agent
        reward = model_p0.postprocess_reward(info)
        
        #split out - really the same for each agent
        if args.printLogs:            # Log step information
            logging.info(f"Step: {num_steps}, Reward: {reward}, Info: {info}")

            # Log observation details
            # logging.info(f"Observation: {obs}")

        # Log player information
        for player in info.get("players", []):
            elixir = player.get("elixir", "N/A")
            hand = player.get("hand", "N/A")
            crowns = player.get("crowns", "N/A")
            logging.info(f"Player {player.get('player_id', 'N/A')} - Elixir: {elixir}, Hand: {hand}, Crowns: {crowns}")

        # Check for negative elixir
        for player in info.get("players", []):
            elixir = player.get("elixir", 0)
            if elixir < 0:
                logging.error(f"Negative elixir detected for player {player.get('player_id', 'N/A')}: {elixir}")
                raise ValueError(f"Negative elixir detected for player {player.get('player_id', 'N/A')}: {elixir}")

        # Append step info to log
        entry = {
            "tick": info.get("tick"),
            "time": info.get("time"),
            "last_action": info.get("last_action"),
            "players": info.get("players"),
            "reward": float(reward),
        }
        if args.printLogs:
            log_fh.write(json.dumps(entry) + "\n")
            log_fh.flush()

        #UPDATE THE SAMPLE_INFO.JSON JUST CUZ
        def to_json_safe(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.integer, np.floating)):
                return x.item()
            if isinstance(x, dict):
                return {k: to_json_safe(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [to_json_safe(v) for v in x]
            return x
        info = to_json_safe(info)
        try:
            with open("replays/sample_info.json", "w") as info_fh:
                json.dump(info, info_fh, indent=4)
        except:
            try:
                with open("replays/sample_info.json", "w") as info_fh:
                    json.dump(info, info_fh, indent=4)
            except Exception as e:
                logging.error(f"Failed to serialize info to JSON for sample_info.json: {e}")
                logging.info("Keeping the existing content of sample_info.json.")
                # just keep going though don't error out
        num_steps += 1


    if args.printLogs:
        logging.info("Game terminated. Exiting the loop.")

    logging.info("Simulation game ended.")

