import torch
from transformers import Wav2Vec2Model

from model.classifier import VoiceClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VoiceDetector:
    def __init__(self, model_path: str):
        # Load wav2vec encoder
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        ).to(DEVICE)
        self.wav2vec.eval()

        # Load trained classifier
        self.classifier = VoiceClassifier().to(DEVICE)
        self.classifier.load_state_dict(
            torch.load(model_path, map_location=DEVICE)
        )
        self.classifier.eval()

    def predict(self, waveform: torch.Tensor) -> float:
        """
        waveform: shape (samples,)
        returns: probability of AI-generated
        """
        waveform = waveform.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embeddings = self.wav2vec(waveform).last_hidden_state
            pooled = embeddings.mean(dim=1)
            score = self.classifier(pooled)

        return score.item()
