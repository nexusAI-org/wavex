from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer
# from src.s2t import *
from src.summarizer import Summarizer
import argparse
import pathlib
import os


class Chain:
    def __init__(self, summarizer, transcriber) -> None:
        self.transcriber = transcriber
        self.summarizer = summarizer

    def summarize_from_audio(self, audio) -> str:
        return ''

def main():
    pass