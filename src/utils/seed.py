import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # benchmark=True for ~10% speedup on fixed input shapes; trade off bit-exact reproducibility
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
