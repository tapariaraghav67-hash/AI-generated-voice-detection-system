import os
import torch
import librosa
from torch.utils.data import Dataset

TARGET_SR = 16000
MAX_DURATION = 5  # seconds
MAX_LEN = TARGET_SR * MAX_DURATION


class VoiceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for label_name, label in [("human", 0), ("ai", 1)]:
            class_dir = os.path.join(root_dir, label_name)

            if not os.path.exists(class_dir):
                raise FileNotFoundError(
                    f"Expected folder not found: {class_dir}"
                )

            for file in os.listdir(class_dir):
                if file.endswith(".wav"):
                    self.samples.append(
                        (os.path.join(class_dir, file), label)
                    )

        if len(self.samples) == 0:
            raise RuntimeError("No audio files found in dataset!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load audio (librosa handles resampling + mono)
        waveform, sr = librosa.load(
            path,
            sr=TARGET_SR,
            mono=True
        )

        # Convert to torch tensor
        waveform = torch.tensor(waveform).float()

        # Safety checks
        if waveform.numel() == 0:
            raise ValueError(f"Empty audio file: {path}")

        if waveform.abs().max() < 1e-4:
            raise ValueError(f"Silent audio file: {path}")

        # Pad / trim
        waveform = waveform[:MAX_LEN]
        if waveform.shape[0] < MAX_LEN:
            pad_len = MAX_LEN - waveform.shape[0]
            waveform = torch.nn.functional.pad(
                waveform, (0, pad_len)
            )

        return waveform, torch.tensor(label, dtype=torch.float32)
