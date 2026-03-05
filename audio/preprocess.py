import torch
import torchaudio

TARGET_SR = 16000
MAX_DURATION = 5  # seconds

def preprocess_audio(waveform, sr):
    waveform = torch.tensor(waveform).float()

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(
            waveform, sr, TARGET_SR
        )

    max_len = TARGET_SR * MAX_DURATION
    waveform = waveform[:max_len]

    if waveform.shape[0] < max_len:
        pad = max_len - waveform.shape[0]
        waveform = torch.nn.functional.pad(
            waveform, (0, pad)
        )

    return waveform
