import math
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr):
        self.opt = optimizer
        self.warmup = warmup_steps
        self.total = total_steps
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup:
            lr = self.base_lr * (self.step_num / max(1, self.warmup))
        else:
            t = (self.step_num - self.warmup) / max(1, (self.total - self.warmup))
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * t))

        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr
