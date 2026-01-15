from inference.wrappers.base import InferenceModel
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.train.ppo_wrapper import PPOObsWrapper


class PPOInferenceModel(InferenceModel):
    def __init__(self, env):
        super().__init__(env)
        
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
        action, _states = self.model.predict(observation, deterministic=True)
        return action
    
    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        obs= self.env.reset()
        return obs