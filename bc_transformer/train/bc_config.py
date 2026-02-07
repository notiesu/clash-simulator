"""
Behavior Cloning configuration.

IMPORTANT CONCEPTUAL NOTE (Option A):
------------------------------------
- The *action space* is ALWAYS size 9:
    0..7 -> deck slot index
    8    -> NOOP

- This is NOT a global card-vocabulary action space.
- The deck can change every game.
- The meaning of slots 0..7 is fixed *within a match*.

- The *input vocabulary* (token embeddings) is global and includes:
    - special tokens (<PAD>, <BOS>, <ME>, <OPP>, etc.)
    - all card IDs seen in the dataset
"""

# =========================
# Reproducibility
# =========================
SEED = 42

# =========================
# Dataset / sequence
# =========================
CTX_LEN = 20          # number of history events to condition on
USE_TIME = True       # whether to include time-delta tokens
TIME_BUCKETS = 16     # number of time-delta buckets

# =========================
# Model architecture
# =========================
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 6
DROPOUT = 0.1
CTX_MAX = 256         # max transformer context length

# =========================
# Action space (FIXED)
# =========================
N_ACTIONS = 9         # 8 deck slots + NOOP (DO NOT CHANGE)

# =========================
# Training
# =========================
BATCH_SIZE = 64
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-2

# =========================
# Scheduler
# =========================
WARMUP_STEPS = 1000

# =========================
# Logging / checkpoints
# =========================
LOG_EVERY = 100
SAVE_EVERY = 1
