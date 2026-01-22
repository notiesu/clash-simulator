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
    def __init__(self, env, printLogs=False):
        self._base_env = env
        self.state = None
        self.episode_start = None
        super().__init__(env, printLogs)

    def wrap_env(self, env):
        self.env = DummyVecEnv([lambda: PPOObsWrapper(env)])
        return self.env

    def reset(self):
        obs = self.env.reset()
        self.state = None
        self.episode_start = np.ones((self.env.num_envs,), dtype=bool)
        return obs

    def load_model(self, model_path):
        self.model = RecurrentPPO.load(model_path, env=self.env)

    def predict(self, obs):
        action, self.state = self.model.predict(
            obs,
            state=self.state,
            episode_start=self.episode_start,
            deterministic=True,
        )
        return action

    def sim_game(self):
        """Run a single episode using the recurrent model. Returns episode reward."""
        log_fh = None
        if hasattr(self, "printLogs") and self.printLogs:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
            logging.info("Starting single-game evaluation.")
            os.makedirs("replays/logs", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"replays/logs/inference_recurrent_{timestamp}.jsonl"
            logging.info(f"Logging actions to {log_path}")
            log_fh = open(log_path, "a")

        obs = self.reset()  # VecEnv reset -> obs only
        self.state = None  # LSTM hidden state
        episode_start = np.ones((self.env.num_envs,), dtype=bool)

        total_reward = 0.0
        steps = 0
        done = False
        infos = None
        win = 0

        if hasattr(self, "printLogs") and self.printLogs:
            print("Starting episode 1/1")

        while not done:
            action, self.state = self.model.predict(
                obs,
                state=self.state,
                episode_start=episode_start,
                deterministic=True,
            )

            obs, reward, done, infos = self.env.step(action)

            try:
                reward_scalar = float(np.asarray(reward)[0])
            except Exception:
                reward_scalar = float(reward)

            total_reward += reward_scalar
            steps += 1

            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos

            if hasattr(self, "printLogs") and self.printLogs:
                print(f"Step: {steps}, Reward: {reward_scalar}")
                # print(f"Observation: {obs}")

            players = info0.get("players", []) if isinstance(info0, dict) else []
            for player in players:
                elixir = player.get("elixir", "N/A")
                hand = player.get("hand", "N/A")
                crowns = player.get("crowns", "N/A")
                if hasattr(self, "printLogs") and self.printLogs:
                    print(
                        f"Player {player.get('player_id', 'N/A')} - Elixir: {elixir}, Hand: {hand}, Crowns: {crowns}"
                    )
                if isinstance(elixir, (int, float)) and elixir < 0:
                    print(f"Negative elixir detected for player {player.get('player_id', 'N/A')}: {elixir}")
                    raise ValueError(f"Negative elixir detected for player {player.get('player_id', 'N/A')}: {elixir}")

            entry = {
                "tick": info0.get("tick") if isinstance(info0, dict) else None,
                "time": info0.get("time") if isinstance(info0, dict) else None,
                "last_action": info0.get("last_action") if isinstance(info0, dict) else None,
                "players": players,
                "reward": reward_scalar,
            }
            if hasattr(self, "printLogs") and self.printLogs and log_fh:
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

            try:
                info_serializable = to_json_safe(info0)
                os.makedirs("replays", exist_ok=True)
                with open("replays/sample_info.json", "w") as info_fh:
                    json.dump(info_serializable, info_fh, indent=4)
            except Exception as e:
                print(f"Failed to serialize info to JSON for sample_info.json: {e}")

        final_info = infos[0] if isinstance(infos, (list, tuple)) else infos
        players = final_info.get("players", []) if isinstance(final_info, dict) else []
        if len(players) >= 2:
            hp_map = {
                p.get("player_id"): sum(
                    float(x) for x in (p.get("king_hp", 0.0), p.get("left_hp", 0.0), p.get("right_hp", 0.0))
                )
                for p in players
            }
            our_hp = hp_map.get(0, 0.0)
            opp_hp = hp_map.get(1, 0.0)
            if our_hp > opp_hp:
                win = 1

        if hasattr(self, "printLogs") and self.printLogs:
            print(f"Episode finished. Reward: {total_reward}, Steps: {steps}, Win: {win}/1")

        if hasattr(self, "printLogs") and self.printLogs and log_fh:
            log_fh.close()

        print(f"Wins: {win}/1")
        print(f"Average episode length: {float(steps):.2f} steps")

        return float(total_reward)
