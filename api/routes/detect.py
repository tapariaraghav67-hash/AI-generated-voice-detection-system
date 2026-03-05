import base64
import io
import os

import librosa
import torch
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from model.inference import VoiceDetector

# -------------------------
# Router
# -------------------------
router = APIRouter()

# -------------------------
# API Key (from env)
# -------------------------
API_KEY = os.getenv("VOICE_API_KEY", "my-hackathon-key-123")

# -------------------------
# Lazy-loaded model
# -------------------------
detector = None

# -------------------------
# Request schema
# -------------------------
class DetectRequest(BaseModel):
    audio_base64: str

# -------------------------
# Detect endpoint
# -------------------------
@router.post("/detect")
def detect_voice(
    req: DetectRequest,
    x_api_key: str = Header(None)
):
    # ---- API key check ----
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # ---- Decode Base64 ----
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # ---- Load audio ----
    try:
        waveform, _ = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,
            mono=True
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio format")

    if waveform is None or len(waveform) == 0:
        raise HTTPException(status_code=400, detail="Empty audio")

    # ---- Convert to tensor ----
    waveform = torch.tensor(waveform).float().unsqueeze(0)

    # ---- Minimum length check (~0.1 sec) ----
    MIN_SAMPLES = 1600
    if waveform.shape[1] < MIN_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail="Audio too short. Provide at least 0.1 seconds."
        )

    # ---- Lazy load model (CRITICAL FIX) ----
    global detector
    if detector is None:
        detector = VoiceDetector("model/model.pt")

    # ---- Predict ----
    score = detector.predict(waveform)

    classification = (
        "AI-generated" if score > 0.5 else "Human-generated"
    )

    explanation = (
        "Synthetic speech patterns detected"
        if score > 0.5
        else "Natural pitch and timing variation detected"
    )

    return {
        "classification": classification,
        "confidence": round(float(score), 3),
        "explanation": explanation
    }
