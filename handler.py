import runpod
import os
import subprocess

def handler(job):
    # Get the input from the job
    input = job.get("input", {})

    # Start logging session using print
    print("Starting training job with input:", input)

    # TODO - add inference capabilities

    # Training params
    train_dir = input.get("train_dir", "train_dir_default")
    run_name = input.get("run_name", "default_run")
    args = []

    for key, value in input.items():
        args.append(f"--{key}")
        args.append(str(value))

    print("Training with args:", args)
    
    runtrain = ["./api-train.sh"] + args

    # ENSURE LOGS ARE UNBUFFERED
    proc = subprocess.run(
        runtrain,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    # This line is only reached if the subprocess actually returned
    print("api-train exited with code", proc.returncode)

    if proc.returncode != 0:
        raise RuntimeError("api-train failed")

# Start the serverless handler
runpod.serverless.start({"handler": handler})
