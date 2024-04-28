import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains.llm import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts.prompt import PromptTemplate
# import torch

class Agent:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer


class Summarizer:
    template = """
                Write a summary of the following text delimited by triple backticks.
                Return your response which covers the key points of the text.
                ```{text}```
                SUMMARY:
            """
    

    def __init__(self, config: dict, model_name: str='') -> None:
        self.config = config
        self.model_name = model_name
        agent = self.get_agent(model_name)
        pipeline = transformers.pipeline(
            'text-generation',
            model=agent.model,
            tokenizer=agent.tokenizer,
            # torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            max_length=config['max_length'],
            do_sample=True,
            top_k=config['top_k'],
            num_return_sequences=config['num_return'],
            pad_token_id=agent.tokenizer.eos_token_id
        )

        llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={
                                  'temperature': config['temperature']})
        prompt = PromptTemplate(template=Summarizer.template, input_variables=["text"])
        self.llm_chain = LLMChain(prompt=prompt, llm=llm)

    def get_agent(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='auto')
        return Agent(model, tokenizer)

    def generate(self, text: str) -> str:
        return self.llm_chain.invoke(text)

