import runpod
import os
import subprocess
from pathlib import Path

def handler(job):
    raw_input = job.get("input", {})
    print("Starting training job with input:", raw_input)

    # Copy so we can pop keys without mutating original
    job_input = dict(raw_input)

    # --- keys NOT meant for train.py ---
    s3_save_uri = job_input.pop("s3_save_uri", None)  # <-- don't forward to train.py
    # (Optional: also support older key name)
    if s3_save_uri is None:
        s3_save_uri = job_input.pop("S3_SAVE_URI", None)

    # Build CLI args for /api-train.sh from remaining keys
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

    # Upload artifacts to S3 (if requested)
    save_path = job_input.get("save_path") or job_input.get("SAVE_PATH")
    if s3_save_uri:
        if not save_path:
            raise RuntimeError("s3_save_uri was provided but save_path is missing.")

        p = Path(str(save_path))
        if not p.exists():
            raise RuntimeError(f"save_path does not exist: {save_path}")

        cmd = ["aws", "s3", "sync", str(save_path), str(s3_save_uri)]
        print("Uploading artifacts:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        return {"ok": True, "return_code": rc, "log_tail": tail, "uploaded_to": s3_save_uri}

    return {"ok": True, "return_code": rc, "log_tail": tail}

runpod.serverless.start({"handler": handler})
