#!/bin/sh

# two recurrent PPO
python -m inference --p0_model_path models/recurrentppo_1lr_checkpoint_3.zip \
    --p1_model_path models/recurrentppo_1lr_checkpoint_7.zip \
    --p0_model_type RecurrentPPO \
    --p1_model_type RecurrentPPO \
    --printLogs

#recurrent PPO vs random policy
python -m inference --p0_model_path models/recurrentppo_1lr_checkpoint_3.zip \
    --p1_model_path na \
    --p0_model_type RecurrentPPO \
    --p1_model_type RandomPolicy \
    --printLogs