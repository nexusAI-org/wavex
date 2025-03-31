from src.stt import WhisperTranscriber
from src.summarizer import Summarizer
import argparse
import pathlib
from pprint import pprint

class Wavex:
    def __init__(self, configs: dict) -> None:
        self.configs = configs
        self.transcriber = WhisperTranscriber(configs.get('language', 'en'), configs.get('chunk_size', 30))        

    def summarize_from_audio(self, audio) -> str:
        transcription = self.transcriber.transcribe(audio)
        with open('transcription.txt', 'w') as f:
            f.write(transcription)
        if not hasattr(self, 'summarizer'):
            if self.configs.get('max_length', None) is None:
                self.configs['max_length'] = 1000
            if self.configs.get('min_length', None) is None:
                self.configs['min_length'] = self.configs['max_length'] // 3
            self.summarizer = Summarizer(self.configs)
        return self.summarizer.generate(transcription)

def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', type=str, default='en')
    parser.add_argument('args', nargs='+', type=str)
    return parser.parse_args()

def main():
    args = retrieve_args()
    language = args.language
    config = {
        'language': language,
        'max_length': 512
    }
    audios = args.args
    rets = []
    wavex = Wavex(config)
    for audio in audios:
        rets.append(wavex.summarize_from_audio(pathlib.Path(audio)))
    pprint(rets)
