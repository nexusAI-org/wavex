import pathlib
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy",
#                   "clean", split="validation")
# sample = ds[0]["audio"]

class WhisperTranscriber:
    def __init__(self, language, chunk_size=30) -> None:
        self.language = language
        self.chunk_size = chunk_size
        self.make_processor()


    def make_processor(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task="transcribe")
    

    def __generate_transcription(self, audio, sampling_rate, duration): #!!!!! TODO: Fix the audio trimming
        transcription = ''
        for i in range(0, int(duration), self.chunk_size):
            start_time = int(np.floor(i * len(audio) / duration))
            end_time = min(i + self.chunk_size, duration)

            # Print chunk boundaries and lengths for debugging
            print(f"Chunk {i+1}: Start - {start_time}, End - {end_time}")

            # Handle last chunk (optional, include remaining audio)
            if end_time == duration:
                chunk_audio = audio[start_time:]
            else:
                chunk_audio = audio[start_time:int(
                    end_time * len(audio) / duration)]

            # Print chunk audio length for debugging
            print(f"Chunk {i+1} Length: {len(chunk_audio)}")

            input_features = self.processor(
                chunk_audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
            predicted_ids = self.model.generate(
                input_features, forced_decoder_ids=self.forced_decoder_ids)
            transcription += self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True)[0] + ' '
        return transcription
    
    def __format_sample(self, sample) -> tuple[np.ndarray, float, int]: 
        audio, original_sr = librosa.load(pathlib.Path(sample))
        resampled_audio = librosa.resample(
            audio, orig_sr=original_sr, target_sr=16000)
        return resampled_audio, 16000, librosa.get_duration(y=resampled_audio)

    def transcribe(self, sample_path: str) -> str:
        return self.__generate_transcription(*self.__format_sample(sample_path))