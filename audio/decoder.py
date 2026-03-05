import base64
import io
import soundfile as sf

def decode_base64_mp3(b64_audio: str):
    audio_bytes = base64.b64decode(b64_audio)
    audio_buffer = io.BytesIO(audio_bytes)
    waveform, sr = sf.read(audio_buffer)
    return waveform, sr
