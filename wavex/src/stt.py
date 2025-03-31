import pathlib
import librosa
import numpy as np
from transformers import pipeline


class WhisperTranscriber:
    def __init__(self, language, chunk_size=30) -> None:
        self.pipeline = pipeline(model='openai/whisper-small', chunk_length_s=chunk_size)#, language=language)
    
    def transcribe(self, audio_path: str) -> str:
        audio, original_sr = librosa.load(pathlib.Path(audio_path))
        resampled_audio = librosa.resample(
            audio, orig_sr=original_sr, target_sr=16000)
        
        return self.pipeline(resampled_audio)['text']
