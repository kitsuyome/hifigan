import os
import torch
import torchaudio

from src.model import Generator
from src.config import MelSpectrogramConfig

CHECKPOINT_PATH = '/kaggle/working/checkpoints/checkpoint.pth'
TEST_PATH = '/kaggle/working/hifigan/src/data/testmels'
OUTPUT_PATH = '/kaggle/working/outs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(checkpoint_path):
    model = Generator()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['generator'])
    model.to(DEVICE)
    model.eval()
    return model

def create_melspec_transform():
    config = MelSpectrogramConfig()
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sr,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        f_min=config.f_min,
        f_max=config.f_max,
        n_mels=config.n_mels,
        power=config.power,
        center=config.center
    )

def generate_audio(generator, test_path, output_path):
    melspec_transform = create_melspec_transform()

    os.makedirs(output_path, exist_ok=True)

    with torch.inference_mode():
        for file in os.listdir(test_path):
            mel_data = torch.load(os.path.join(test_path, file)).to(DEVICE)
            generated_wav = generator(mel_data).squeeze(0).detach().cpu()

            if generated_wav.ndim == 1:
                generated_wav = generated_wav.unsqueeze(0)

            torchaudio.save(os.path.join(output_path, file.replace('.pt', '.wav')), generated_wav, 22050)

if __name__ == '__main__':
    generator = load_model(CHECKPOINT_PATH)
    generate_audio(generator, TEST_PATH, OUTPUT_PATH)
