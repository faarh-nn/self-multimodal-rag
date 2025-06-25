import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class QAChain:
    def __init__(self, model_type):
        """
        Question Answering Chain: Baseline without RAG (direct model prompting without additional context)
        The steps are as follows:
        1. Create a QA prompt using the question and an instruction to answer the question.
        2. Call the LLM and prompt it with the created prompt
        3. Obtain the generated answer from the model
        
        :param model_type: The type of model used for answer generation.
        """
        print(f"Using OpenAI model: {model_type}")
        self.model_type = model_type
        self.model = ChatOpenAI(
            model=model_type,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=400
        )

        self.tokenizer = None
        
        # here we don't have a retriever since we prompt the model directly, without any additional context
        self.chain = (self.prompt_func | self.model | StrOutputParser())
        
        
    def run(self, inputs):
        return self.chain.invoke(inputs)
    

    def prompt_func(self, data_dict):
        
        qa_prompt = """Anda adalah asisten AI yang ahli menjawab pertanyaan tentang sistem neraca nasional.\n"""
        
        # Construct the prompt
        messages = [
            SystemMessage(content=qa_prompt),  # Context system instruction
            HumanMessage(content=data_dict['question'])  # User question
        ]
        
        return messages