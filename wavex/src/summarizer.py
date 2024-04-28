import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains.llm import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts.prompt import PromptTemplate
import torch

class Summarizer:
    template = """
                Write a summary of the following text delimited by triple backticks.
                Return your response which covers the key points of the text.
                ```{text}```
                SUMMARY:
            """
    
    def __init__(self, config: dict, model_id: str ='Falconsai/text_summarization') -> None:
        self.config = config
        self.model_id = model_id
        self.pipeline = transformers.pipeline(
            'summarization',
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_length=config['max_length'],
            min_length=config['min_length'],
            do_sample=False
        )

    def generate(self, text: str) -> str:
        return self.pipeline(text)[0]['summary_text']

