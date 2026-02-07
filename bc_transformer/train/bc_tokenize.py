import math

# ============================================================
# Token constants (MUST stay consistent across repo)
# ============================================================
PAD = "<PAD>"
BOS = "<BOS>"
ME  = "<ME>"
OPP = "<OPP>"

# Action constant used in data (and slot-policy action_list)
NOOP = "NOOP"


# ============================================================
# Optional helper: time-delta bucketing
# (You can use this later if you want the model to see timing)
# ============================================================
def bucket_time_delta(dt, n_buckets: int = 16, dt_max: float = 10.0) -> int:
    """
    Log-ish bucketing so tiny deltas are separated better.
    dt in seconds; cap at dt_max.
    returns int bucket in [0, n_buckets-1]
    """
    dt = max(0.0, min(float(dt), float(dt_max)))
    # map [0, dt_max] -> [0,1], then log-scale
    x = dt / dt_max if dt_max > 0 else 0.0
    y = math.log1p(9 * x) / math.log1p(9)  # 0..1
    b = int(y * (n_buckets - 1) + 1e-9)
    return max(0, min(n_buckets - 1, b))
