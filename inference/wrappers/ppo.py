from inference.wrappers.base import InferenceModel
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.train.ppo_wrapper import PPOObsWrapper


class PPOInferenceModel(InferenceModel):
    def __init__(self, env, printLogs=False):
        super().__init__(env, printLogs)

    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        return self.env.reset()
    
    def wrap_env(self, env):
        """
        Wrap the environment as needed for the model.
        This method can be overridden by subclasses if specific wrapping is required.
        """
        self.env = DummyVecEnv([lambda: PPOObsWrapper(env)])
        return self.env

    def load_model(self, model_path):
        """
        Load the model from the specified path.
        This method should be implemented by subclasses.
        """
        self.model = PPOInferenceModel(self.env)  # pass base env
        self.load_model(model_path)
    
    def predict(self, observation):
        """
        Perform inference using the loaded model.
        This method should be implemented by subclasses.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model before prediction.")
        return self.model.predict(observation)[0]
        
    def load_model(self, model_path):
        """
        Load the PPO model from the specified path.
        """
        # Implement model loading logic here
        self.model= PPO.load(model_path, env=self.env)

    def predict(self, observation):
        """
        Perform inference using the loaded PPO model.
        """
        # Implement prediction logic here
        return self.model.predict(observation, deterministic=True)[0]
    