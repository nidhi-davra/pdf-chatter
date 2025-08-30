import numpy as np


_whisper_model = None


def get_whisper_model(model_name: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        import whisper  # lazy import
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model


def record_audio(duration: int = 5, samplerate: int = 16000) -> np.ndarray:
    import sounddevice as sd  # lazy import
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def transcribe_audio_array(audio: np.ndarray) -> str:
    model = get_whisper_model()
    result = model.transcribe(audio)
    return result.get("text", "").strip()


