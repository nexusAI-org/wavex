import transformers
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.mapreduce import MapReduceChain


class Summarizer:

    map_prompt_template = """
                      Write a summary of this chunk of text that includes the main points and any important details.
                      {text}
                      """
    map_prompt = PromptTemplate(
        template=map_prompt_template, input_variables=["text"])
    combine_prompt_template = """
                        Write a concise summary of the following text delimited by triple backquotes.
                        Return your response in bullet points which covers the key points of the text.
                        ```{text}```
                        BULLET POINT SUMMARY:
                        """
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text"]
    )

    def __init__(self, config: dict, model_id: str = 'google-bert/bert-base-uncased') -> None:
        self.config = config
        if self.config.get('chunk_overlap', None) is None:
            self.config['chunk_overlap'] = 50
        if self.config.get('chunk_size', None) is None:
            self.config['chunk_size'] = 800
        self.model_id = model_id
        pipeline = transformers.pipeline(
            'text-generation',
            model=model_id,
            torch_dtype=torch.bfloat16,
            max_new_tokens=config['max_length'],
            do_sample=False,
        )
        self.pipeline = HuggingFacePipeline(pipeline=pipeline)


    def generate(self, text: str) -> str:
        chunks = RecursiveCharacterTextSplitter(separators=[' '],
            chunk_size=self.config['chunk_size'], chunk_overlap=self.config['chunk_overlap']).split_text(text)
        
        docs = [Document(page_content=chunk) for chunk in chunks]

        
        summarizer = load_summarize_chain(self.pipeline, chain_type='map_reduce', map_prompt=Summarizer.map_prompt,
                                          combine_prompt=Summarizer.combine_prompt,
                                          return_intermediate_steps=False,)
        ret = summarizer.run(docs)
        print(ret)
        # return ' '.join([self.pipeline(chunk)[0]['summary_text'] for chunk in chunks])

if __name__ == '__main__':
    with open('transcription.txt', 'r') as f:
        text = f.read()
    test = Summarizer({'max_length': 512, 'min_length': 128}).generate(text)
    print(test)