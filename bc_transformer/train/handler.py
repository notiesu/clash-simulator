import runpod
import os
import subprocess
from pathlib import Path

def _aws_sync(local_dir: str, s3_uri: str):
    # Make sure local exists
    p = Path(local_dir)
    if not p.exists():
        raise RuntimeError(f"save_path does not exist: {local_dir}")

    # Sync directory to S3
    cmd = ["aws", "s3", "sync", local_dir, s3_uri]
    print("Uploading artifacts:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def handler(job):
    job_input = job.get("input", {})
    print("Starting training job with input:", job_input)

    # Build CLI args for /api-train.sh
    args = []
    for key, value in job_input.items():
        args.append(f"--{key}")
        args.append(str(value))

    runtrain = ["/api-train.sh"] + args
    print("Running:", " ".join(runtrain))

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    proc = subprocess.Popen(
        runtrain,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
        universal_newlines=True,
    )

    output_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)

    rc = proc.wait()
    print("api-train exited with code", rc)

    tail = "".join(output_lines[-200:])

    if rc != 0:
        raise RuntimeError(f"api-train failed (rc={rc}). Last logs:\n{tail}")

    # After successful training, upload to S3 if requested
    save_path = job_input.get("save_path") or job_input.get("SAVE_PATH")
    s3_save_uri = job_input.get("s3_save_uri") or job_input.get("S3_SAVE_URI")

    if s3_save_uri:
        if not save_path:
            # If user forgot to pass save_path, default to something safe
            save_path = "/tmp/runs"
        _aws_sync(str(save_path), str(s3_save_uri))
        return {
            "ok": True,
            "return_code": rc,
            "log_tail": tail,
            "uploaded_to": s3_save_uri,
        }

    return {"ok": True, "return_code": rc, "log_tail": tail}

runpod.serverless.start({"handler": handler})
