from abc import ABC, abstractmethod


class EvaluatorInterface(ABC):
    """
    Abstract base class for an LLM Evaluator.
    """

    @abstractmethod
    def run_evaluation(self) -> dict:
        """
        Perform evaluation and return results as a dictionary.
        """
        pass

    @abstractmethod
    def get_prompt(self, inputs: dict):
        """
        Construct the prompt for evaluation.
        """
        pass