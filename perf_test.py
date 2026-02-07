from wrappers.recurrentppo import RecurrentPPOInferenceModel
from src.clasher.gym_env import ClashRoyaleGymEnv
import time

if __name__ == "__main__":
    model_path = "models/recurrentppo_1lr_checkpoint_3.zip"  # Update with your model path
    model = RecurrentPPOInferenceModel(model_path, eval=True, deterministic=True)
    model.reset()
    #generate an obs from the env
    env = ClashRoyaleGymEnv()
    obs, _ = env.reset()

    model.perf_test(model.preprocess_observation(obs))

    # vs non eval mode
    model2 = RecurrentPPOInferenceModel(model_path, eval=False, deterministic=True)
    model2.reset()

    model2.perf_test(model2.preprocess_observation(obs))
