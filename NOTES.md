Typical learning rate range for PPO: 1e-4 → 5e-4

helloworld.py - basic environment stepping, log random games
    -usage: python helloworld.py
    -game output: replays/logs

play_replay.py - visualize game in pygame
    -usage: python play_replay.py {GAME_LOG_PATH}
    
TODOs:
Refactor inference code with proper abstractions
Troubleshoot model
    - tweak reward to take into account elixir, hand, stuff like that
GPU infra
    - docker base image containing environment and engine
    - handler image for entrypoint training scripts
    - allow uploading training packages
    
Issues:
Pathfinding and retargeting are broken
Tower range is slightly off

INSTRUCTIONS FOR TRAINING:
Create a directory containing training package, including all code and data. 
This package MUST contain a train.py as this will be the entrypoint for the server to execute.
This train.py must allow for this cmdline argument: --output_dir
S.t. the server can call it like this: python train.py --output_dir "output"
You can also add more arguments if you would like, just make sure to pass it in the HTTP request to runpod and the handler will unpack all the arguments with the double-tag just as in the format above.

IMPORTANT NOTE: MAKE SURE PACKAGE IMPORTS ARE DONE CORRECTLY! STRUCTURE YOUR ENVIRONMENT LIKE THIS:
-project root
    - training package
        -train.py
        -data
        -wrappers and other needed training code
    - clash-simulator
        -src
            -gym_env.py
    - output

and try to run your train.py like this: python -m ${INSERT_TRAINING_DIR_HERE}.train --output_dir "output". If you get output and no errors then it should work properly on the server!
Note this doesn't handle external dependencies for right now. We can add something to make sure external packages are installed correctly next time, or you can just add the packages to your training package.


