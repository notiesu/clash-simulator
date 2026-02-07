from wrappers.recurrentppo import RecurrentPPOInferenceModel
from wrappers.rppo_onnx import RecurrentPPOONNXInferenceModel
from wrappers.rppowrappers import PPOObsWrapper, PPORewardWrapper
from src.clasher.gym_env import ClashRoyaleGymEnv
import time

if __name__ == "__main__":
    model_path = "models/recurrentppo_1lr_checkpoint_3.zip"  # Update with your model path
    onnx_path = "models/recurrentppo.onnx"
    model = RecurrentPPOInferenceModel(model_path, eval=True, deterministic=True)
    model.reset()


    #generate an obs from the env
    env = ClashRoyaleGymEnv()
    obs, _ = env.reset()

    #onnx mode
    model_onnx = RecurrentPPOONNXInferenceModel(onnx_path)
    model_onnx.reset()
    model_onnx.perf_test(model_onnx.preprocess_observation(obs))

    #eval mode

    obs = model.preprocess_observation(obs)
    model.perf_test(obs)

    # vs non eval mode
    model2 = RecurrentPPOInferenceModel(model_path, eval=False, deterministic=True)
    model2.reset()

    model2.perf_test(model2.preprocess_observation(obs))
