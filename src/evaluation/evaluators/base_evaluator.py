import os
from abc import abstractmethod
from langchain.chains.transform import TransformChain
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import BooleanOutputParser
from src.evaluation.evaluators.evaluator_interface import EvaluatorInterface
from langchain.output_parsers.openai_tools import PydanticToolsParser
import time
import random


class EvaluationResult(BaseModel):
    """The result of an evaluation for a given metric"""

    grade: str = Field(description="the grade after evaluating the metric (YES or NO)")
    reason: str = Field(description="The reasoning behind the grading decision")


class BaseEvaluator(EvaluatorInterface):
    """  
    A base class for an LLM evaluator using direct OpenAI API..
  
    Attributes: 
        model (str): The model to be used for evaluation.
        model_type (AzureChatOpenAI or LlavaNextForConditionalGeneration): Type of the model to use for evaluation.
        json_parser (JsonOutputParser): Parser used to parse evaluation results to a json object.
        boolean_parser (BooleanOutputParser): Parser used to parse the assigned grade to a boolean value.
        check_grade_chain (TransformChain): Applies the transformation from the LLM output for the grade to a boolean value.
        fix_format_parser (OutputFixingParser): Parser used to fix misformatted json output of an LLM.
    """
    def __init__(self, model, **kwargs):
        """  
        Initializes the BaseEvaluator object.
  
        :param model_name: The OpenAI model name to be used for evaluation (e.g., "gpt-4o-mini").
        :param temperature: Controls randomness in the model's output (0 = deterministic, 1 = creative).:param model: The model to be used for evaluation.
        
        Keyword Args:
            user_query (str): The user query
            generated_answer (str): The answer produced by the model
            reference_answer (str): The ground truth answer
            context (str): The texts retrieved by the retrieval system
            image (str): The image retrieved by the retrieval system
        """
        # self.model = ChatOpenAI(
        #     model=model_type,
        #     api_key=os.getenv("OPENAI_API_KEY"),
        #     temperature=temperature,
        #     max_tokens=500
        # )
        self.model = model
        self.json_parser = JsonOutputParser(pydantic_object=EvaluationResult)
        self.boolean_parser = BooleanOutputParser()
        self.kwargs = kwargs
        self.check_grade_chain = TransformChain(
            input_variables=["grade", "reason"],
            output_variables=["grade", "reason"],
            transform=self.get_numeric_score
        )
        
        # Initialize the fixing parser using the same model
        self.tools_parser = PydanticToolsParser(tools=[EvaluationResult])
        

    def get_numeric_score(self, inputs: str) -> dict:
        """
        Checks that the obtained grade (YES or NO) can be parsed to a boolean and sets the grade to its integer value (0 or 1)
        """
        inputs["grade"] = int(self.boolean_parser.parse(inputs["grade"]))
        return inputs

    def run_evaluation(self, max_retries = 10) -> dict:
        """  
        Performs evaluation for one output of a RAG system.
        Creates an evaluation chain that constructs the prompt, calls the model, fixes possible 
        json formatting errors and checks the validity of the assigned grade.

        :return: A json object with a grade (0 or 1) and a reason for the grading as string.
        """ 
        retries = 0
        while retries < max_retries:
            try:
                chain = RunnableLambda(self.get_prompt) | self.model | self.json_parser | self.check_grade_chain
                result = chain.invoke(self.kwargs)
                return result
            except Exception as e:
                # Cek apakah error terkait rate limit
                if "rate_limit_exceeded" in str(e) and retries < max_retries:
                    # Hitung waktu tunggu dengan exponential backoff
                    wait_time = (2 ** retries) + random.random()
                    print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds... ({retries+1}/{max_retries})")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    print(f"Standard JSON parsing failed: {e}. Trying with tools parser...")
                    # Fallback to tools parser which works better with newer models
                    try:
                        chain = RunnableLambda(self.get_prompt) | self.model | self.tools_parser | self.check_grade_chain
                        result = chain.invoke(self.kwargs)
                        return result
                    except Exception as e2:
                        if "rate_limit_exceeded" in str(e2) and retries < max_retries:
                            # Hitung waktu tunggu dengan exponential backoff
                            wait_time = (2 ** retries) + random.random()
                            print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds... ({retries+1}/{max_retries})")
                            time.sleep(wait_time)
                            retries += 1
                        else:
                            print(f"All parsing methods failed: {e2}")
                            # Return a default result in case of failure
                            return {"grade": 0, "reason": f"Evaluation failed due to parsing error: {str(e2)}"}

    @abstractmethod
    def get_prompt(self, inputs: dict):
        """
        Construct the prompt for evaluation based on a dictionary containing required input arguments.
        """
        pass