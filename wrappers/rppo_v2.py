import logging
import os
import datetime
import json
from xml.parsers.expat import model
import numpy as np
import torch
from src.clasher.model import InferenceModel
from src.clasher.card_encoder import CardEncoder
from src.clasher.data import CardDataLoader
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from scripts.train.ppo_wrapper import PPOObsWrapper
from wrappers.recurrentppo import RecurrentPPOInferenceModel, RewardWrapper, PPO
import time
# try common locations for RecurrentPPO

# reuse the same observation wrapper used by PPO inference
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

class PPO_V2(PPO):
    def __init__(self, 
                 board_shape=(3, 18, 32),
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
                 device="cpu",
                 preferred_placements=None,
                 placement_alpha=0.3):
        super().__init__(
            board_shape=board_shape,
            num_elixir=num_elixir,
            num_hand_slots=num_hand_slots,
            num_card_ids=num_card_ids,
            num_cycle_slots=num_cycle_slots,
            cnn_output_dim=cnn_output_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            num_actions=num_actions,
            use_gru=use_gru,
            lr=lr,
            device=device
        )


        default_preferred_placements = {
            "HogRider": (1, 17),
            "Cannon": (8, 22),
            "Musketeer": (8, 31),
            "IceGolemite": (2, 17),
            "IceSpirit": (2, 17),
            "Skeletons": (9, 17),
            "Fireball": (3, 9),
            "Log": (3, 17),
        }


        self.preferred_placements = preferred_placements if preferred_placements is not None else default_preferred_placements

        #build heatmap for per-card priors
        self.placement_priors = torch.zeros((num_card_ids, self.W, self.H), device=self.device)

        self.card_encoder = CardEncoder(CardDataLoader())
        for i, (card_name, (x, y)) in enumerate(self.preferred_placements.items()):
            card_id = self.card_encoder.encode(card_name)
            if card_id is not None and card_id < num_card_ids:
                heatmap = self.build_gaussian_heatmap(self.W, self.H, x, y, sigma=1.5, device=self.device)
                mirrored_heatmap = self.build_gaussian_heatmap(self.W, self.H, self.W-1-x, y, sigma=1.5, device=self.device)
                self.placement_priors[card_id] = heatmap + mirrored_heatmap
                self.placement_priors[card_id] /= self.placement_priors[card_id].sum()  # normalize

        self.placement_alpha = placement_alpha

        #flatten heatmap to match placement head output
        self.placement_priors = self.placement_priors.view(num_card_ids, -1)

    def build_gaussian_heatmap(self, W, H, center_x, center_y, sigma=1.5, device="cpu"):
        xs = torch.arange(W, device=device).view(W,1).expand(W,H)
        ys = torch.arange(H, device=device).view(1,H).expand(W,H)
        heatmap = torch.exp(-((xs - center_x)**2 + (ys - center_y)**2)/(2*sigma**2))
        heatmap /= heatmap.sum()  # normalize to sum=1
        return heatmap
        
    def act(self, obs, hidden_states=None, deterministic=False, valid_action_mask=None, play_bias=0.0):
        #PPO_V2 biases the placement logits based on self.placement_priors
        #valid action mask is passed as tensor by predict

        device = self.device

        # move obs tensors
        obs = {k: v.to(device) for k, v in obs.items()}

        if valid_action_mask is not None:
            valid_action_mask = valid_action_mask.to(device)

        
        if valid_action_mask is not None:
             # slice off NO_OP if it’s at the end
            mask_no_op = valid_action_mask[:, -1:]  # optional
            mask_cards = valid_action_mask[:, :-1]  # [1, 2304]
            # reshape to [1, num_hand_slots, W*H]
            valid_action_mask = mask_cards.view(1, self.num_hand_slots, self.W * self.H)
        
        # Forward pass
        logits, value, hidden_states = self.forward(obs, hidden_states)
        
        play_logits = logits["play"] + play_bias  # only apply bias in training
        
        # ---- Play vs NO-OP ----
        play_dist = Categorical(logits=play_logits)
        if deterministic:
            play_action = torch.argmax(play_logits, dim=-1)
        else:
            play_action = play_dist.sample()
        log_prob = play_dist.log_prob(play_action)
        entropy = play_dist.entropy()

        
        # ---- Card selection ----
        card_logits = logits["card"]
        if valid_action_mask is not None:
            valid_card_mask = valid_action_mask[:, :, :self.num_hand_slots].any(dim=2)  # [1, num_hand_slots]
            inf_mask = torch.where(
                valid_card_mask,
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
        
        # ---- Placement selection ----
        placement_logits = logits["placement"]


        #prepare card action - card idx -> card name -> card id
        hands = obs["hands"][:, 0, :]  # (B, 4)
        card_ids = torch.gather(hands, 1, card_action.unsqueeze(1)).squeeze(1)  # (B,)

        #bias by the placement priors for the selected card
        self.placement_priors = self.placement_priors.to(device)  # do this once in init ideally
        priors = self.placement_priors[card_ids]
        eps = 1e-8
        biased_placement_logits = placement_logits + self.placement_alpha * torch.log(priors + eps)  # broadcast over batch

        if valid_action_mask is not None:
            #use card action to index into mask
            valid_placement_mask = valid_action_mask[torch.arange(valid_action_mask.size(0)), card_action]
            inf_mask = torch.where(
                valid_placement_mask,
                torch.zeros_like(biased_placement_logits),
                torch.full_like(biased_placement_logits, -1e8)
            )
            biased_placement_logits = biased_placement_logits + inf_mask

        placement_dist = Categorical(logits=biased_placement_logits)
        if deterministic:
            placement_action = torch.argmax(biased_placement_logits, dim=-1)
        else:
            placement_action = placement_dist.sample()
        placement_log_prob = placement_dist.log_prob(placement_action)
        placement_entropy = placement_dist.entropy()
        
        # Only count card & placement if play_action == 1
        play_mask = (play_action == 1).float()
        log_prob += play_mask * (card_log_prob + placement_log_prob)
        entropy += play_mask * (card_entropy + placement_entropy)
        
        #TODO - I dont think is needed since they wont be part of the logprob update, but keeping just in case
        # Set NO_OP for card/placement when not playing
        # NO_OP_CARD = self.num_hand_slots  # or whatever you defined
        # NO_OP_PLACEMENT = self.placement_head.out_features  # use a sentinel if needed
        # card_actions = torch.where(play_action == 1, card_action, NO_OP_CARD)
        # placement_actions = torch.where(play_action == 1, placement_action, NO_OP_PLACEMENT)
        
        actions = (play_action, card_action, placement_action)
        
        return actions, log_prob, value, hidden_states
    
class RecurrentPPOInferenceModel_V2(RecurrentPPOInferenceModel):
    def __init__(self, device, model_path=None, eval=False, deterministic=False):
        super().__init__(model_path=model_path, eval=eval, deterministic=deterministic)
        self.model = PPO_V2(
            board_shape=(3, 18, 32),
            num_elixir=2,
            num_hand_slots=4,
            num_card_ids=128,
            num_cycle_slots=4,
            cnn_output_dim=256,
            mlp_hidden_dim=128,
            gru_hidden_dim=128,
            num_actions=2305,
            use_gru=True,
            device=device,
            placement_alpha=0.3
            )
        #some custom parameters for reward shaping

        self.load_model(model_path)
        
        self.episode_start = None
       
       
        self.model.eval()

        
        #some custom parameters for reward shaping
        self.model.to(self.device)