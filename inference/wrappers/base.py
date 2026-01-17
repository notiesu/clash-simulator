import logging
import os
import datetime
import json

class InferenceModel:
    def __init__(self, env, printLogs=False):
        self.env = self.wrap_env(env)
        self.model = None
        self.printLogs = printLogs
        self.lastObs = None

    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        raise NotImplementedError("Subclasses must implement the load_model method.")
        
    
    def wrap_env(self, env):
        """
        Wrap the environment as needed for the model.
        This method can be overridden by subclasses if specific wrapping is required.
        """
        return env

    def load_model(self, model_path):
        """
        Load the model from the specified path.
        This method should be implemented by subclasses.
        """

        raise NotImplementedError("Subclasses must implement the load_model method.")
    
    def predict(self, observation):
        """
        Perform inference using the loaded model.
        This method should be implemented by subclasses.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model before prediction.")
        raise NotImplementedError("Subclasses must implement the predict method.")
    
    def sim_game(self):
        """
        Internal method to run the environment loop.
        This can be overridden by subclasses if specific loop behavior is needed.
        """
        obs = self.reset()
        done = False
        num_steps = 0

        if hasattr(self, 'printLogs') and self.printLogs:
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
            action = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            #TODO - maybe fix? little bit of a hack in case info returns as a list
            info = info if isinstance(info, dict) else info[0]  # Handle VecEnv info format

            if hasattr(self, 'printLogs') and self.printLogs:
                logging.info(f"Step: {num_steps}, Reward: {reward}, Info: {info}")

                # Log observation details
                logging.info(f"Observation: {obs}")

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
            if hasattr(self, 'printLogs') and self.printLogs:
                log_fh.write(json.dumps(entry) + "\n")
                log_fh.flush()

        if hasattr(self, 'printLogs') and self.printLogs:
            logging.info("Game terminated. Exiting the loop.")
