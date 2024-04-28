# WAVEX

Audio summarizer AI 

## INSTALLATION

To install wavex, use the following commands:
```sh
git clone https://github.com/varad-comrad/wavex
cd wavex
pip install -r requirements.txt
```

All's set! Now, you can use it like this:
```sh
python wavex.py <audio_file> 
```

Additionally, if you wish to use wavex as a library, all you have to do is:


## ARCHITECTURE

Wavex was built on top of Whisper, the OpenAI speech-to-text model. It works in 2 steps: first, transcribing the input from the audio file. Then, a second model, chosen by the user (default ) is called to summarize the text. 