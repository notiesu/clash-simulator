import runpod
import os
import subprocess
import logging

def handler(job):
    # Get the input from the job
    input = job.get("input", {})

    #start logging session
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting training job with input: %s", input)

    #TODO - add inference capabilities

    #training params
    train_dir = input.get("train_dir", "train_dir_default")
    run_name = input.get("run_name", "default_run")
    args = []

    for key, value in input.items():
        args.append(f"--{key}")
        args.append(str(value))

    logging.info("Training with args: %s", args)
    
    runtrain = ["./api-train.sh"] + args

    # Call the voice cloning script
    subprocess.run(runtrain, check=True)

# Start the serverless handler
runpod.serverless.start({"handler": handler})
