import torch
from dataclasses import dataclass, asdict
from typing import Tuple
from src.utils import transform_dict

@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    center: bool = False
    pad_value: float = -11.5129251

@dataclass
class ExperimentConfig:
    n_epochs: int = 200
    batch_size: int = 16
    lr: float = 0.0002
    sched_decay: float = 0.999
    betas: Tuple[float, float] = (0.8, 0.99)
    segment_length: int = 8192
    l1_gamma: float = 45.
    matching_gamma: float = 2.
    adv_gamma: float = 1.
    save_epochs: int = 2
    seed: int = 1234
    num_workers: int = 4
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    save_dir: str = '/kaggle/working'
    data_dir: str = '/kaggle/working/hifigan/src/data/LJSpeech-1.1/wavs'
    test_path: str = '/kaggle/input/testmels'
    project_name: str = 'dla-hw-4'
    checkpoint_path: str = '/kaggle/input/checkpoint/checkpoint_6ep.pth'

    def to_dict(self, expand: bool = True):
        return transform_dict(asdict(self), expand)