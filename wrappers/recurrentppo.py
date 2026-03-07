import logging
import os
import datetime
import json
from xml.parsers.expat import model
import numpy as np
import torch
from src.clasher.model import InferenceModel
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from scripts.train.ppo_wrapper import PPOObsWrapper
import time
# try common locations for RecurrentPPO

# reuse the same observation wrapper used by PPO inference
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        info0 = info[0] if isinstance(info, (list, tuple)) else info
        shaped_reward = 0.0

        # base reward if present in info, otherwise 0
        base_reward = float(info0.get("reward", 0.0)) if isinstance(info0, dict) else 0.0
        shaped_reward += base_reward

        # gather current tower HPs
        players = info0.get("players", []) if isinstance(info0, dict) else []
        cur_hps = {}
        for p in players:
            pid = p.get("player_id")
            cur_hps[pid] = (
                float(p.get("king_hp", 0.0)),
                float(p.get("left_hp", 0.0)),
                float(p.get("right_hp", 0.0)),
            )

        # init trackers if missing
        if self._prev_tower_hps is None:
            self._prev_tower_hps = cur_hps.copy()
            self._prev_time = float(info0.get("time", 0.0)) if isinstance(info0, dict) else 0.0
            self._prev_elixir_waste = float(info0.get("elixir_waste", 0.0)) if isinstance(info0, dict) else 0.0
            return obs, shaped_reward, terminated, truncated, info

        # 1) Defensive Tower Health Reward
        r_tower = 0.0
        for pid in cur_hps:
            prev = self._prev_tower_hps.get(pid, (0.0, 0.0, 0.0))
            cur = cur_hps[pid]
            for i in range(3):
                prev_hp = prev[i]
                cur_hp = cur[i]
                delta_h = max(0.0, prev_hp - cur_hp)
                H = self._H_main if i == 0 else self._H_aux
                sign = (-1) ** (pid + 1)
                r_tower += sign * (delta_h / H)
        shaped_reward += r_tower

        # 2) Defensive Tower Destruction Reward
        r_destroy = 0.0
        for pid in cur_hps:
            prev = self._prev_tower_hps.get(pid, (0.0, 0.0, 0.0))
            cur = cur_hps[pid]
            for i in range(3):
                prev_hp = prev[i]
                cur_hp = cur[i]
                if prev_hp > 0.0 and cur_hp <= 0.0:
                    base = 3.0 if i == 0 else 1.0
                    sign = (-1) ** (pid + 1)
                    r_destroy += sign * base
        shaped_reward += r_destroy

        # 3) Main Tower Activation Reward
        r_activate = 0.0
        for pid in cur_hps:
            prev = self._prev_tower_hps.get(pid, (0.0, 0.0, 0.0))
            cur = cur_hps[pid]
            aux_prev_alive = prev[1] > 0.0 and prev[2] > 0.0
            aux_cur_alive = cur[1] > 0.0 and cur[2] > 0.0
            if aux_prev_alive and aux_cur_alive and not self._main_hit_seen.get(pid, False) and cur[0] < prev[0]:
                sign = (-1) ** pid
                r_activate += sign * 0.1
                self._main_hit_seen[pid] = True
        shaped_reward += r_activate

        # 4) Elixir Overflow Penalty: -0.1 per full second of continued overflow
        cur_elixir_waste = float(info0.get("elixir_waste", 0.0)) if isinstance(info0, dict) else 0.0
        cur_time = float(info0.get("time", self._prev_time)) if isinstance(info0, dict) else self._prev_time
        if cur_elixir_waste > self._prev_elixir_waste + 1e-9:
            self._elixir_overflow_accum += (cur_time - self._prev_time)

        penalty = 0.0
        if self._elixir_overflow_accum >= 1.0:
            n = int(self._elixir_overflow_accum)
            penalty = 0.1 * n
            self._elixir_overflow_accum -= n

        shaped_reward -= penalty

        # update trackers
        self._prev_tower_hps = cur_hps.copy()
        self._prev_elixir_waste = cur_elixir_waste
        self._prev_time = cur_time

        return obs, float(shaped_reward), terminated, truncated, info
    

class PPO(nn.Module):
    def __init__(self, 
                 board_shape=(3, 128, 128),
                 num_elixir=2,
                 num_hand_slots=4,
                 num_card_ids=127,
                 num_cycle_slots=4,
                 cnn_output_dim=256,
                 mlp_hidden_dim=128,
                 gru_hidden_dim=128,
                 num_actions=2305,
                 use_gru=True,
                 lr=1e-4,
                 device="cpu"):
        super().__init__()
        self.device = device
        self.use_gru = use_gru
        self.num_hand_slots = num_hand_slots
        self.num_cycle_slots = num_cycle_slots

        # CNN for board
        self.hand_embed = nn.Embedding(num_card_ids+1, 16, device=self.device)  # +1 for unknown, +1 for padding
        self.cycle_embed = nn.Embedding(num_card_ids+1, 8, device=self.device)  # +1 for unknown, +1 for padding
        C, H, W = board_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*H*W, cnn_output_dim),
            nn.ReLU()
        )

        # Embeddings for hand and cycle
        self.unknown_idx = num_card_ids  # index for unknown cards
        
        

        # MLP for global features + hand/cycle embeddings
        self.num_hand_cards = 4
        self.num_cycle_cards = 4
        mlp_input_dim = num_elixir + self.num_hand_cards * self.hand_embed.embedding_dim * 2 + self.num_cycle_cards * self.cycle_embed.embedding_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU()
        )

        # Optional GRU for recurrence
        self.gru_hidden_dim = gru_hidden_dim
        if use_gru:
            self.gru = nn.GRU(cnn_output_dim + mlp_hidden_dim, gru_hidden_dim, batch_first=True)
            feature_dim = gru_hidden_dim
        else:
            feature_dim = cnn_output_dim + mlp_hidden_dim

        # Actor & Critic heads
        self.policy_net = nn.Linear(feature_dim, num_actions)
        self.play_head = nn.Linear(feature_dim, 2)          # Level 1: NO-OP vs Play
        self.value_net = nn.Linear(feature_dim, 1)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    def act(self, obs, hidden_states=None, deterministic=False, valid_action_mask=None):

        logits, value, hidden_states = self.forward(obs, hidden_states)

        play_logits = logits["play"]
        card_logits = logits["action"]

        # ---- Play vs NOOP ----
        play_dist = Categorical(logits=play_logits)

        if deterministic:
            play_action = torch.argmax(play_logits, dim=-1)
        else:
            play_action = play_dist.sample()

        log_prob = play_dist.log_prob(play_action)
        entropy = play_dist.entropy()

        # ---- Card selection ----
        if valid_action_mask is not None:
            inf_mask = torch.where(
                valid_action_mask,
                torch.zeros_like(card_logits),
                torch.full_like(card_logits, -1e8)
            )
            card_logits = card_logits + inf_mask

        card_dist = Categorical(logits=card_logits)

        if deterministic:
            card_action = torch.argmax(card_logits, dim=-1)
        else:
            card_action = card_dist.sample()

        card_log_prob = card_dist.log_prob(card_action)
        card_entropy = card_dist.entropy()

        play_mask = (play_action == 1).float()

        log_prob += card_log_prob * play_mask
        entropy += card_entropy * play_mask

        NO_OP = 2304
        card_actions = torch.where(play_action == 0, NO_OP, card_action)

        actions = (play_action, card_actions)

        return actions, log_prob, value, hidden_states
    
    def obs_to_tensor(self, obs, device="cpu"):
        board = torch.tensor(obs["board"], dtype=torch.float32, device=device)
        # check dims
        if board.ndim == 3:  # single env: (H,W,C)
            board = board.permute(2,0,1).unsqueeze(0) / 255.0  # (1,C,H,W)
        elif board.ndim == 4:  # already batched: (B,H,W,C)
            board = board.permute(0,3,1,2).float() / 255.0      # (B,C,H,W)
        else:
            raise ValueError(f"Unexpected board shape: {board.shape}")

        # elixirs
        elixirs = torch.tensor(obs["elixirs"], dtype=torch.float32, device=device)
        if elixirs.ndim == 1:
            elixirs = elixirs.unsqueeze(0)  # (1,2)
        # hands
        hands = torch.tensor(obs["hands"], dtype=torch.long, device=device)
        if hands.ndim == 2:
            hands = hands.unsqueeze(0)  # (1,2,4)
        # cycles
        cycles = torch.tensor(obs["cycles"], dtype=torch.long, device=device)
        if cycles.ndim == 2:
            cycles = cycles.unsqueeze(0)  # (1,2,4)

        return {"board": board, "elixirs": elixirs, "hands": hands, "cycles": cycles}

    def forward(self, obs, hidden_states=None):

        """
        obs: tensorized observation dict with keys 'board', 'elixirs', 'hands', 'cycles'
        hidden_state: optional (1, B, H) for GRU
        """
        board = obs['board']
        elixirs = obs['elixirs'].float()  # ensure elixirs are float

        # hand embeddings

        hands = obs['hands']  # (B, num_hand_slots)
        hands = torch.where(hands < 0, torch.tensor(self.unknown_idx, device=hands.device), hands)
        hands = torch.clamp(hands, 0, self.hand_embed.num_embeddings - 1)  # shape (B, num_hand_slots)
        hand_emb = self.hand_embed(hands).flatten(1)  # (B, slots, embed_dim)

        # cycle embeddings
        cycles = obs['cycles']  # (B, num_cycle_slots)
        cycles = torch.where(cycles < 0, torch.tensor(self.unknown_idx, device=cycles.device), cycles)
        cycles = torch.clamp(cycles, 0, self.cycle_embed.num_embeddings - 1)  # shape (B, num_cycle_slots)
        cycle_emb = self.cycle_embed(cycles).flatten(1)

        # CNN + MLP
        cnn_feat = self.cnn(board)
        #need to see elixirs + hand_emb + cycle_emb size
        # print(f"elixirs: {elixirs.shape}, hand_emb: {hand_emb.shape}, cycle_emb: {cycle_emb.shape}")
        mlp_input = torch.cat([elixirs, hand_emb, cycle_emb], dim=1)
        mlp_feat = self.mlp(mlp_input)

        combined = torch.cat([cnn_feat, mlp_feat], dim=1).unsqueeze(1)  # (B, 1, dim) for GRU

        # GRU
        if self.use_gru:
            gru_out, hidden_states = self.gru(combined, hidden_states)
            features = gru_out.squeeze(1)
        else:
            features = combined.squeeze(1)

        # Actor & Critic
        action_logits = self.policy_net(features)
        play_logits = self.play_head(features)
        values = self.value_net(features).squeeze(-1)  # (B,)

        return {"action": action_logits, "play": play_logits}, values, hidden_states

    def export_onnx(self, filepath):
        device = next(self.parameters()).device
        dummy_obs = {
            'board': torch.zeros(1, 3, 128, 128, dtype=torch.float32, device=device),
            'elixirs': torch.zeros(1, 2, dtype=torch.float32, device=device),
            'hands': torch.zeros(1, self.num_hand_slots, dtype=torch.long, device=device),
            'cycles': torch.zeros(1, self.num_cycle_slots, dtype=torch.long, device=device),
        }
        dummy_hidden = None
        if self.use_gru:
            dummy_hidden = torch.zeros(1, 1, self.gru_hidden_dim, device=device)  # (num_layers * num_directions, batch, hidden_size)

        torch.onnx.export(
            self,
            (dummy_obs, dummy_hidden),
            filepath,
            input_names=['obs', 'hidden_states'] if self.use_gru else ['obs'],
            output_names=['action_logits', 'value', 'new_pi_h', 'new_pi_c', 'new_vf_h', 'new_vf_c'] if self.use_gru else ['action_logits', 'value'],
            dynamic_axes={
                'obs': {0: 'batch_size'},
                'hidden_states': {1: 'batch_size'} if self.use_gru else None,
                'action_logits': {0: 'batch_size'},
                'value': {0: 'batch_size'},
                'new_pi_h': {1: 'batch_size'} if self.use_gru else None,
                'new_pi_c': {1: 'batch_size'} if self.use_gru else None,
                'new_vf_h': {1: 'batch_size'} if self.use_gru else None,
                'new_vf_c': {1: 'batch_size'} if self.use_gru else None,
            },
            opset_version=11
        )
        print(f"Model exported to {filepath}")

class RecurrentPPOInferenceModel(InferenceModel):
    def __init__(self, model_path=None, eval=False, deterministic=False):
         # record flags
        self.eval_mode = eval
        self.deterministic = deterministic

         # Eval-mode optimizations
        if self.eval_mode:
            # set policy to eval and disable grad globally for inference
            self.model.eval()
            torch.set_grad_enabled(False)

            # prefer GPU + fp16 when available for faster inference
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                try:
                    self.model.to(self.device)
                except Exception:
                    logging.exception("Failed to move model to CUDA; continuing on current device")
                # use automatic mixed precision for faster kernels when supported
                self.use_autocast = True
                # enable cudnn benchmark for fixed-size ops
                try:
                    torch.backends.cudnn.benchmark = True
                except Exception:
                    pass
            else:
                self.device = torch.device("cpu")
                self.use_autocast = False

            if hasattr(self.model.policy, "lstm"):
                self.model.policy.lstm.flatten_parameters()
        else:
            # not eval: default device still CPU
            self.device = torch.device("cpu")
            self.use_autocast = False

        self.model = PPO(
            board_shape=(3, 128, 128),
            num_elixir=2,
            num_hand_slots=4,
            num_card_ids=128,
            num_cycle_slots=4,
            cnn_output_dim=256,
            mlp_hidden_dim=128,
            gru_hidden_dim=128,
            num_actions=2305,
            use_gru=True,
            )
        self.model.to(self.device)
        self.load_model(model_path)
        self.episode_start = None
        #some custom parameters for reward shaping

       

       

    def reset(self):
        pass
    
    def load_model(self, model_path):
        if model_path is None:
            return
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def predict(self, obs, valid_action_mask=None, state=None):
        if self.eval_mode:
            # Use inference_mode to skip autograd overhead. If CUDA is available
            # and use_autocast is True, enable AMP for faster kernel execution.
            with torch.inference_mode():
                if self.use_autocast and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        action, _,_, next_state = self.model.act(
                            obs,
                            state,
                            self.deterministic,
                            valid_action_mask=valid_action_mask
                        )
                else:
                    #instead of predict, use forward
                    action, _,_, next_state = self.model.act(
                        obs,
                        state,
                        self.deterministic,
                        valid_action_mask=valid_action_mask
                    )
        else:
            action, _,_, next_state = self.model.act(
                obs,
                state,
                self.deterministic,
                valid_action_mask=valid_action_mask
            )
        # Ensure next_state is defined consistently when using eval_mod
        # Validate predicted action against mask if provided; be robust to different mask types
        # print(next_state)
        return action[1]

    def perf_test(self, obs):
        N = 500
        start = time.perf_counter()

        
        for _ in range(N):
            self.predict(obs)

        end = time.perf_counter()
        print (f"RecurrentPPO inference time for {N} steps in eval_mode={self.eval_mode}: {end - start:.4f} seconds")


    def preprocess_observation(self, observation):
        # Normalize/convert HWC -> CHW (and NHWC -> NCHW) for numpy inputs and dict {"p1-view": HWC}
        return self.model.obs_to_tensor(observation, device=self.device)

    def postprocess_action(self, model_output, agent_id=None):
        # For this env the model_output can be passed through directly.
        return model_output

    def postprocess_reward(self, info):
        # No reward shaping for now, just return the base reward from info
        if isinstance(info, (list, tuple)):
            info = info[0]  # handle case where info is a list of dicts for vectorized envs
        return float(info.get("reward", 0.0)) if isinstance(info, dict) else 0.0