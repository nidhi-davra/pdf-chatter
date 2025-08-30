import whisper
import sounddevice as sd
import numpy as np

model = whisper.load_model("base")

def record_and_transcribe(duration=5, samplerate=16000):
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    print("✅ Recording finished, transcribing...")

    # Flatten to 1-D float32 for Whisper (avoids ffmpeg dependency)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    result = model.transcribe(audio)
    print("📝 Text:", result["text"])

# Example: record for 5 seconds
record_and_transcribe(5)
