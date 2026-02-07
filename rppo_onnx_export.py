import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
import gymnasium as gym


# ---------- CONFIG ----------
MODEL_PATH = "models/recurrentppo_1lr_checkpoint_7.zip"
ONNX_PATH = "recurrentppo.onnx"
DEVICE = "cpu"
DETERMINISTIC = True
OPSET = 14
N_ENVS = 1
# ----------------------------


# ---------- LOAD MODEL ----------
model = RecurrentPPO.load(MODEL_PATH, device=DEVICE)
policy = model.policy
policy.eval()


# ---------- DUMMY OBS ----------
obs_space = model.observation_space

if len(obs_space.shape) == 3:
    # CNN policy
    C, H, W = obs_space.shape
    dummy_obs = torch.zeros((N_ENVS, C, H, W), dtype=torch.float32)
else:
    # MLP policy
    dummy_obs = torch.zeros((N_ENVS, obs_space.shape[0]), dtype=torch.float32)

dummy_episode_starts = torch.zeros((N_ENVS,), dtype=torch.bool)


# ---------- DUMMY LSTM STATES ----------
lstm_shape = policy.lstm_hidden_state_shape  # (n_layers, n_envs, hidden_size)

dummy_states = RNNStates(
    pi=(
        torch.zeros(lstm_shape),
        torch.zeros(lstm_shape),
    ),
    vf=(
        torch.zeros(lstm_shape),
        torch.zeros(lstm_shape),
    ),
)


# ---------- WRAPPER ----------
class RecurrentPPOWrapper(nn.Module):
    def __init__(self, policy, deterministic=True):
        super().__init__()
        self.policy = policy
        self.deterministic = deterministic

    def forward(
        self,
        obs,
        pi_h, pi_c,
        vf_h, vf_c,
        episode_starts,
    ):
        lstm_states = RNNStates(
            pi=(pi_h, pi_c),
            vf=(vf_h, vf_c),
        )

        actions, _, _, new_states= self.policy.forward(
            obs,
            lstm_states,
            episode_starts,
            deterministic=self.deterministic,
        )

        return (
            actions,
            new_states.pi[0], new_states.pi[1],
            new_states.vf[0], new_states.vf[1],
        )


wrapper = RecurrentPPOWrapper(policy, DETERMINISTIC).to(DEVICE)
wrapper.eval()


# ---------- EXPORT ----------
torch.onnx.export(
    wrapper,
    (
        dummy_obs,
        dummy_states.pi[0], dummy_states.pi[1],
        dummy_states.vf[0], dummy_states.vf[1],
        dummy_episode_starts,
    ),
    ONNX_PATH,
    opset_version=OPSET,
    input_names=[
        "obs",
        "pi_h", "pi_c",
        "vf_h", "vf_c",
        "episode_starts",
    ],
    output_names=[
        "actions",
        "new_pi_h", "new_pi_c",
        "new_vf_h", "new_vf_c",
    ],
    dynamic_axes={
        "obs": {0: "batch"},
        "episode_starts": {0: "batch"},
        "actions": {0: "batch"},
        "pi_h": {1: "batch"},
        "pi_c": {1: "batch"},
        "vf_h": {1: "batch"},
        "vf_c": {1: "batch"},
        "new_pi_h": {1: "batch"},
        "new_pi_c": {1: "batch"},
        "new_vf_h": {1: "batch"},
        "new_vf_c": {1: "batch"},
    },
)

print(f"ONNX export successful → {ONNX_PATH}")
