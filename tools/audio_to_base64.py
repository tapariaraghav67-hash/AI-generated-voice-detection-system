import base64
import json

AUDIO_PATH = "data/val/human/your_sample.wav"

with open(AUDIO_PATH, "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "audio_base64": encoded
}

# Print VALID JSON (important)
print(json.dumps(payload))


