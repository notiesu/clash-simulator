import os
import time
import subprocess
from pathlib import Path
import runpod


def env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if (v is not None and str(v).strip() != "") else default


def handler(job):
    # ---- run metadata / output dir ----
    run_name = env("RUN_NAME", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    volume_dir = env("VOLUME_DIR", "/runpod-volume")
    out_dir = Path(env("OUT_DIR", str(Path(volume_dir) / "runs" / run_name)))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- dataset paths (these should be local paths inside the container) ----
    train_jsonl = env("TRAIN_JSONL", str(Path(volume_dir) / "data"))
    val_jsonl   = env("VAL_JSONL",   str(Path(volume_dir) / "data"))

    # ---- training knobs ----
    task        = env("TASK", "joint")          # card | gate | joint
    epochs      = env("EPOCHS", "10")
    batch_size  = env("BATCH_SIZE", "64")
    lr          = env("LR", "1e-4")
    history_len = env("HISTORY_LEN", "20")
    device      = env("DEVICE", "cuda")

    lambda_gate = env("LAMBDA_GATE", "1.0")
    lambda_card = env("LAMBDA_CARD", "1.0")

    save_path = Path(env("SAVE_PATH", str(out_dir / "bc_model.pt")))

    # ---- build the exact train command ----
    cmd = [
        "/api-train.sh",
        "--task", task,
        "--train_jsonl", train_jsonl,
        "--val_jsonl", val_jsonl,
        "--device", device,
        "--lr", lr,
        "--epochs", epochs,
        "--batch_size", batch_size,
        "--history_len", history_len,
        "--save_path", str(save_path),
    ]

    # only matters for joint task, but safe to pass always
    cmd += ["--lambda_gate", lambda_gate, "--lambda_card", lambda_card]

    print("=== BC handler starting ===")
    print(f"RUN_NAME     = {run_name}")
    print(f"OUT_DIR      = {out_dir}")
    print(f"TRAIN_JSONL  = {train_jsonl}")
    print(f"VAL_JSONL    = {val_jsonl}")
    print(f"TASK         = {task}")
    print(f"DEVICE       = {device}")
    print(f"SAVE_PATH    = {save_path}")
    print("CMD:", " ".join(cmd))

    # ---- dataset download (optional) ----
    skip_download = os.environ.get("SKIP_DOWNLOAD", "").strip().lower() in ("1", "true", "yes", "on")
    s3_uri = os.environ.get("S3_URI", "").strip()

    if skip_download:
        print("SKIP_DOWNLOAD=1 -> skipping dataset download")
    elif s3_uri:
        subprocess.run(["/workspace/download_data.sh"], check=True)
    else:
        print("S3_URI not set -> skipping dataset download (expect data already in TRAIN_JSONL/VAL_JSONL)")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
