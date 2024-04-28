from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy",
#                   "clean", split="validation")
# sample = ds[0]["audio"]

class WhisperTranscriber:
    def __init__(self, laguage) -> None:
        self.language
        self.make_processor()


    def make_processor(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task="transcribe")
    

    def transcribe(self, audio):
        # input_features = processor(
        #     sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
        # predicted_ids = model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids)
        # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        # [' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']        
        pass