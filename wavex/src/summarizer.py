import transformers
import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


class Summarizer:
    def __init__(self, config: dict, model_id: str ='Falconsai/text_summarization') -> None:
        self.config = config
        if self.config.get('chunk_overlap', None) is None:
            self.config['chunk_overlap'] = 20
        if self.config.get('chunk_size', None) is None:
            self.config['chunk_size'] = 1024
        self.model_id = model_id
        pipeline = transformers.pipeline(
            'summarization',
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_length=config['max_length'],
            min_length=config['min_length'],
            do_sample=False,
        )
        self.pipeline = HuggingFacePipeline(pipeline=pipeline)

    def generate(self, text: str) -> str:
        chunks = CharacterTextSplitter(separator=' ',
            chunk_size=self.config['chunk_size'], chunk_overlap=self.config['chunk_overlap']).split_text(text)
        print(chunks)
        return ' '.join([self.pipeline(chunk)[0]['summary_text'] for chunk in chunks])

if __name__ == '__main__':
    test = Summarizer({'max_length': 512, 'min_length': 128}).generate('this is a test')