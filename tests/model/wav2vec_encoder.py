from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch

class Wav2VecEncoder:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )
        self.model.eval()

    def encode(self, waveform):
        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state
