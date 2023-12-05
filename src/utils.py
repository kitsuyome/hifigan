import random
import os
import torch
from torch import nn

class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_value = 0
        self.avg = 0
        self.total_sum = 0
        self.updates_count = 0

    def update(self, value, count=1):
        self.current_value = value
        self.total_sum += value * count
        self.updates_count += count
        self.avg = self.total_sum / self.updates_count

def init_weights(module, mean=0.0, std=0.01):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        module.weight.data.normal_(mean, std)

def calc_padding(kernel_size, dilation):
    return (kernel_size * dilation - dilation) // 2

def collate_fn(samples):
    return (
        torch.cat([s[0] for s in samples], 0),
        torch.cat([s[1] for s in samples], 0)
    )

def transform_dict(config_dict, expand = True):
    ret = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            t = transform_dict(dict(enumerate(v)), expand)
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True