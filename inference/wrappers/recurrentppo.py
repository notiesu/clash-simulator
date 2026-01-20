import logging
import os
import datetime
import json
import numpy as np
from inference.wrappers.base import InferenceModel
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from scripts.train.ppo_wrapper import PPOObsWrapper

# filepath: /Users/fcp/Desktop/CLASHROYALE/clash-simulator/inference/wrappers/recurrentppo.py


# try common locations for RecurrentPPO

# reuse the same observation wrapper used by PPO inference


class RecurrentPPOInferenceModel(InferenceModel):
    """
    Inference wrapper for recurrent PPO models (sb3_contrib.RecurrentPPO).
    Handles recurrent states and masks during inference and provides a
    custom sim_game loop that keeps RNN states in sync with environment done flags.
    """
    def __init__(self, env, printLogs=False):
        self._base_env = env
        self.states = None
        self.masks = None  # mask used by recurrent policies (per-env)
        super().__init__(env, printLogs)

    def wrap_env(self, env):
        # Single-env DummyVecEnv with the same observation wrapper used for PPO
        self.env = DummyVecEnv([lambda: PPOObsWrapper(env)])
        return self.env

    def reset(self):
        """
        Reset the wrapped environment and clear recurrent states/masks.
        Returns the initial observation (VecEnv format).
        """
        obs = self.env.reset()
        self.states = None
        self.masks = None
        return obs

    def load_model(self, model_path):
        """
        Load the RecurrentPPO model from disk.
        """
        if RecurrentPPO is None:
            raise ImportError("RecurrentPPO (sb3_contrib) not available in the environment.")
        # load model with the wrapped env
        self.model = RecurrentPPO.load(model_path, env=self.env)

    def predict(self, observation):
        """
        Predict action using the recurrent model. Tries to pass recurrent states and masks.
        Falls back to the non-recurrent predict signature if needed.
        Returns the selected action (not the state tuple).
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model before prediction.")
        try:
            # Many sb3_contrib recurrent policies expect (obs, state, mask)
            action, self.states = self.model.predict(
                observation, state=self.states, mask=self.masks, deterministic=True
            )
        except TypeError:
            # Fallback if the model.predict signature does not accept state/mask
            action = self.model.predict(observation, deterministic=True)[0]
            # keep states/masks as-is
        return action

    def sim_game(self):
        """
        Run the environment loop using the recurrent-aware predict method so that
        RNN hidden states are reset when episodes end.
        This mirrors the logging behavior from the base implementation.
        """
        obs = self.reset()
        done = False
        num_steps = 0

        if hasattr(self, 'printLogs') and self.printLogs:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
            logging.info("Environment reset. Starting the recurrent game loop.")
            os.makedirs("replays/logs", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"replays/logs/inference_recurrent_{timestamp}.jsonl"
            logging.info(f"Logging actions to {log_path}")
            log_fh = open(log_path, "a")

        while not done:
            action = self.predict(obs)
            obs, reward, done, info = self.env.step(action)

            # Handle VecEnv formats
            # done may be array-like for VecEnv; normalize to array
            done_arr = np.asarray(done) if isinstance(done, (list, tuple, np.ndarray)) else np.asarray([done])
            # masks: True where not done (keep state), False where done (reset next step)
            # Use boolean mask; some implementations accept floats as well
            self.masks = (~done_arr).astype(bool)

            # If single-env, simplify done and info
            if isinstance(done, (list, tuple, np.ndarray)):
                done_scalar = bool(done_arr.any())
            else:
                done_scalar = bool(done)

            # Info normalization
            info = info if isinstance(info, dict) else info[0]

            if hasattr(self, 'printLogs') and self.printLogs:
                logging.info(f"Step: {num_steps}, Reward: {reward}, Info: {info}")
                logging.info(f"Observation: {obs}")

            for player in info.get("players", []):
                elixir = player.get("elixir", "N/A")
                hand = player.get("hand", "N/A")
                crowns = player.get("crowns", "N/A")
                logging.info(f"Player {player.get('player_id', 'N/A')} - Elixir: {elixir}, Hand: {hand}, Crowns: {crowns}")

            for player in info.get("players", []):
                elixir = player.get("elixir", 0)
                if elixir < 0:
                    logging.error(f"Negative elixir detected for player {player.get('player_id', 'N/A')}: {elixir}")
                    raise ValueError(f"Negative elixir detected for player {player.get('player_id', 'N/A')}: {elixir}")

            entry = {
                "tick": info.get("tick"),
                "time": info.get("time"),
                "last_action": info.get("last_action"),
                "players": info.get("players"),
                "reward": float(reward),
            }
            if hasattr(self, 'printLogs') and self.printLogs:
                log_fh.write(json.dumps(entry) + "\n")
                log_fh.flush()

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

            info_json = to_json_safe(info)
            try:
                with open("replays/sample_info.json", "w") as info_fh:
                    json.dump(info_json, info_fh, indent=4)
            except Exception as e:
                logging.error(f"Failed to serialize info to JSON for sample_info.json: {e}")

            num_steps += 1
            done = done_scalar

        if hasattr(self, 'printLogs') and self.printLogs:
            logging.info("Recurrent game terminated. Exiting the loop.")