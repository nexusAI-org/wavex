from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer
from src.s2t import WhisperTranscriber
from src.summarizer import Summarizer
import argparse
import pathlib
import os


class Wavex:
    def __init__(self, summarizer: Summarizer, transcriber: WhisperTranscriber) -> None:
        self.transcriber = transcriber
        self.summarizer = summarizer

    def summarize_from_audio(self, audio) -> str:
        return self.summarizer.generate(self.transcriber.transcribe(audio))

def main():
    

    Wavex(
        Summarizer(

        ),
        WhisperTranscriber(

        )
    ).summarize_from_audio()