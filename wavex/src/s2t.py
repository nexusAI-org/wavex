from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy",
#                   "clean", split="validation")
# sample = ds[0]["audio"]

class WhisperTranscriber:
    def __init__(self, laguage) -> None:
        self.language = laguage
        self.make_processor()


    def make_processor(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task="transcribe")
    

    def __generate_trasncription(self, sample):
        input_features = self.processor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
        predicted_ids = self.model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    def __format_sample(self, sample): 
        pass

    def __format_output(self, output) -> str:
        pass

    def transcribe(self, sample) -> str:
        return self.__format_output(self.__generate_trasncription(self.__format_sample(sample)))