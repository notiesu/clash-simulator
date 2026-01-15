# train_ppo.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.clasher.gym_env import ClashRoyaleGymEnv
import gymnasium as gym
import numpy as np
from ppo_wrapper import PPOObsWrapper

# Create logs and model dirs
os.makedirs("replays/logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# Wrap env for PPO
env = PPOObsWrapper(ClashRoyaleGymEnv())
vec_env = DummyVecEnv([lambda: env])

# Instantiate PPO

#TODO - this doesn't run games to completion due to the step limiting. Fix by training on GPU + increasing steps?
model = PPO(
    "CnnPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    policy_kwargs=dict(normalize_images=False)
)

# Train for 100k timesteps (adjust as needed)
model.learn(total_timesteps=100_000, progress_bar=True)

# Save model
model.save("models/ppo_clashroyale_baseline")
print("Training finished, model saved.")
