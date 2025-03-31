import transformers
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate


class Summarizer:

    prompt_template = """
                      Write a summary of this chunk of text that includes the main points and any important details.
                      {text}
                      """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["text"])
    sysnthesis_prompt_template = """
                        Write a concise summary out of of the following summaries delimited by triple backquotes.
                        Return a single cohesive and concise summary that maintains clarity and coherence.
                        
                        ```
                        {summaries}
                        ```
                        
                        SUMMARY:
                        """
    sysnthesis_prompt = PromptTemplate(
        template=sysnthesis_prompt_template, input_variables=["summaries"]
    )

    def __init__(self, config: dict, model_id: str = 't5-small') -> None:
        self.config = config
        if self.config.get('chunk_overlap', None) is None:
            self.config['chunk_overlap'] = 50
        if self.config.get('chunk_size', None) is None:
            self.config['chunk_size'] = 512
        self.model_id = model_id
        pipeline = transformers.pipeline(
            'summarization',
            model=model_id,
            torch_dtype=torch.bfloat16,
            max_length=config.get('max_length', 128),
            min_length=config.get('min_length', 32),
            truncation=True,
        )
        self.pipeline = HuggingFacePipeline(pipeline=pipeline)


    def generate(self, text: str) -> str:
        chunks = RecursiveCharacterTextSplitter(separators=[' '],
            chunk_size=self.config['chunk_size'], chunk_overlap=self.config['chunk_overlap']).split_text(text)
        
        docs = [Document(page_content=chunk) for chunk in chunks]

        out = []
        summarizer = Summarizer.prompt | self.pipeline 
        for doc in docs:
            out.append(summarizer.invoke({'text': doc.page_content}))
        summaries = ('\n' + '-' * 50 + '\n').join(out)
        
        summarizer = Summarizer.sysnthesis_prompt | self.pipeline
        ret = summarizer.invoke({'summaries': summaries})
        return ret, summaries

if __name__ == '__main__':
    with open('transcription.txt', 'r') as f:
        text = f.read()
    test, summaries = Summarizer({'max_length': 512, 'min_length': 128}).generate(text)
    with open('summary.txt', 'w') as f:
        f.write(test)
        
    with open('summaries.txt', 'w') as f:
        f.write(summaries)