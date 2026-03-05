import torch
import librosa

from model.inference import VoiceDetector

# Load trained detector
detector = VoiceDetector("model/model.pt")

# Load audio using librosa (NO torchaudio)
waveform, sr = librosa.load(
    "data/val/human/your_sample.wav",  # <-- change path if needed
    sr=16000,
    mono=True
)

waveform = torch.tensor(waveform).float()

score = detector.predict(waveform)

label = "AI-generated" if score > 0.5 else "Human-generated"

print("Prediction:", label)
print("Confidence:", round(score, 3))
