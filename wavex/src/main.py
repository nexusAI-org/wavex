from src.s2t import WhisperTranscriber
from src.summarizer import Summarizer
import argparse
import pathlib
from pprint import pprint


config = {
    "max_length": 3000,
    'min_length': 30
}

class Wavex:
    def __init__(self, summarizer: Summarizer, transcriber: WhisperTranscriber) -> None:
        self.transcriber = transcriber
        self.summarizer = summarizer

    def summarize_from_audio(self, audio) -> str:
        transcription = self.transcriber.transcribe(audio)
        pprint(transcription)
        return self.summarizer.generate(transcription)

def retrieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', type=str, default='en')
    parser.add_argument('args', nargs='+', type=str)
    return parser.parse_args()

def main():
    args = retrieve_args()
    language = args.language
    audios = args.args
    rets = []
    wavex = Wavex(Summarizer(config), WhisperTranscriber(language))
    for audio in audios:
        rets.append(wavex.summarize_from_audio(pathlib.Path(audio)))
    pprint(rets)
