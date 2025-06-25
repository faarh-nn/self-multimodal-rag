import base64
import importlib
import os
from typing import Dict, List
from langchain_openai import ChatOpenAI

  
class EvaluationModule:
    """
    A class to evaluate the performance of a RAG pipeline based on different LLM-based metrics.
    The models that can be used as evaluators are GPT-4V, LLaVA or CogVLM2.
    The metrics used are Answer Correctness, Answer Relevancy, Text Context Relevancy, Image Context Relevancy,
    Text Faithfulness and Image Faithfulness.
    For each metric, an evaluator with the desired model is created to create the prompt and evaluate the corresponding metric.        
    """
    
    def __init__(self, model_type: str):
        """
        Initializes the EvaluationModule with selected model as evaluator.
  
        :param model_type: model_type (str): Model to use for evaluation (either gpt4_vision or llava).
        """
        self.model_type = model_type
          
        self.model = ChatOpenAI(
            model = model_type,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            max_tokens=500
        )
        
        self.tokenizer=None
        evaluator_module_name = "openai"
          
        # Dynamically import evaluator module based on model_type  
        self.evaluator_module = self._import_evaluator_module(evaluator_module_name)

        # Define supported metrics and required input arguments
        self._metrics = {
            'Answer Relevancy': {
                'eval_method': self._evaluate_answer_relevancy,
                'required_args': ['query', 'generated_answer']
            },
            'Answer Correctness': {
                'eval_method': self._evaluate_answer_correctness,
                'required_args': ['query', 'generated_answer', 'reference_answer']
            },
            'Image Faithfulness': {
                'eval_method': self._evaluate_image_faithfulness,
                'required_args': ['query', 'generated_answer', 'image']
            },
            'Text Faithfulness': {
                'eval_method': self._evaluate_text_faithfulness,
                'required_args': ['query', 'generated_answer', 'context']
            },
            'Image Context Relevancy': {
                'eval_method': self._evaluate_image_context_relevancy,
                'required_args': ['query', 'image']
            },
            'Text Context Relevancy': {
                'eval_method': self._evaluate_text_context_relevancy,
                'required_args': ['query', 'context']
            }
        }
        
    def _import_evaluator_module(self, model_type: str):  
        # Construct module name and import it  
        module_name = f"src.evaluation.evaluators.evaluators_{model_type}"   
        evaluator_module = importlib.import_module(module_name)  
        return evaluator_module  
  
    # Method to dynamically create an evaluator instance  
    def create_evaluator_instance(self, evaluator_class_name: str, **kwargs):  
        # Access the evaluator class from the dynamically imported module  
        evaluator_class = getattr(self.evaluator_module, evaluator_class_name)  
        # Instantiate the evaluator with arguments it needs  
        if self.tokenizer:
            evaluator_instance = evaluator_class(model=self.model, tokenizer=self.tokenizer, **kwargs)
        else:
            evaluator_instance = evaluator_class(model=self.model, **kwargs)
        return evaluator_instance  
  

    # create evaluators for each metric
    def _evaluate_answer_relevancy(self, query: str, generated_answer: str) -> dict:  
        evaluator = self.create_evaluator_instance('AnswerRelevancyEvaluator', user_query=query, generated_answer=generated_answer)  
        return {'Answer Relevancy': evaluator.run_evaluation()}

    def _evaluate_answer_correctness(self, query: str, generated_answer: str, reference_answer: str) -> dict:
        evaluator = self.create_evaluator_instance('AnswerCorrectnessEvaluator', user_query=query, generated_answer=generated_answer, reference_answer=reference_answer)
        return {'Answer Correctness': evaluator.run_evaluation()}
    
    def _evaluate_image_faithfulness(self, query: str, generated_answer: str, image: str) -> dict:
        evaluator = self.create_evaluator_instance('ImageFaithfulnessEvaluator', user_query=query, generated_answer=generated_answer, image=image)
        return {'Image Faithfulness': evaluator.run_evaluation()}
    
    def _evaluate_text_faithfulness(self, query: str, generated_answer: str, context: str) -> dict:
        evaluator = self.create_evaluator_instance('TextFaithfulnessEvaluator', user_query=query, generated_answer=generated_answer, context=context)
        return {'Text Faithfulness': evaluator.run_evaluation()}
    
    def _evaluate_image_context_relevancy(self, query: str, image: str) -> dict:
        evaluator = self.create_evaluator_instance('ImageContextRelevancyEvaluator', user_query=query, image=image)
        return {'Image Context Relevancy': evaluator.run_evaluation()}

    def _evaluate_text_context_relevancy(self, query: str, context: str) -> dict:
        evaluator = self.create_evaluator_instance('TextContextRelevancyEvaluator', user_query=query, context=context)
        return {'Text Context Relevancy': evaluator.run_evaluation()}


    def evaluate(self, metrics: List[str], **kwargs) -> Dict[str, dict]:
        """
        Evaluates the specified metrics and returns the results as a dictionary.

        Args:
            metrics (List): List of metrics to be calculated.

        Keyword Args:
            query (str): The user query
            generated_answer (str): The answer produced by the model
            reference_answer (str): The ground truth answer
            context (str): The texts retrieved by the retrieval system
            image (str): The image retrieved by the retrieval system

        Returns:
            Dict[str, dict]: A dictionary with metric names as keys
            and a dictionary with grade and reason as values.

        Raises:
            ValueError: If a required argument for a metric is missing or if an invalid metric is specified.
        """
        results = {}
        for metric in metrics:
            if metric in self._metrics:
                required_args = self._metrics[metric]['required_args']
                if self._check_required_arguments(required_args, metric, list(kwargs.keys())):
                    metric_kwargs = {arg: kwargs[arg] for arg in required_args}
                    results.update(self._metrics[metric]['eval_method'](**metric_kwargs))
            else:
                raise ValueError(f"Invalid metric '{metric}'\n"
                                 f"Valid metrics: {list(self._metrics.keys())}")
        return results

    def _check_required_arguments(self, required_args: List[str], metric: str, kwargs: List[str]) -> bool:
        """
           Check if all required arguments for a specific metric are present in the provided arguments.

           Args:
               required_args (List[str]): A list of arguments required for the metric.
               metric (str): The name of the metric being checked.
               kwargs (List[str]): A list of provided arguments.

           Returns:
               bool: True if all required arguments are present, False otherwise.

           Raises:
               ValueError: If one or more required arguments are missing.
           """
        missing_args = [arg for arg in required_args if arg not in kwargs]
        if missing_args:
            raise ValueError(f"Missing required arguments for metric '{metric}':\n"
                             f"Required arguments: {self._metrics[metric]['required_args']}\n"
                             f"Missing: {', '.join(missing_args)}")
        return True


if __name__ == '__main__':
    evaluator_model = "gpt-4o-mini"
    evaluation_module = EvaluationModule(evaluator_model)

    user_query = "What is the transformer architecture?"
    context = """The dominant sequence transduction models are based on complex recurrent or convolutional neural
            networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder
            through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely
            on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine
            translation tasks show these models to be superior in quality while being more parallelizable and requiring
            significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation
            task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014
            English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of
            41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from
            the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to
            English constituency parsing both with large and limited training data."""

    img_path = "./img/transformer.PNG"
    with open(img_path, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode("utf-8")

    reference_answer = """The transformer architecture is a popular deep learning model used in natural language
    processing tasks. It replaces recurrent neural networks with a self-attention mechanism, allowing the model to
    capture long-range dependencies more effectively. It consists of an encoder and decoder, each with multiple layers.
    The transformer has achieved state-of-the-art performance in NLP and serves as the basis for models like BERT, GPT,
    and T5."""
    generated_answer = "I love cats"
    
    METRICS = ['Answer Correctness', 'Answer Relevancy','Image Faithfulness',
               'Image Context Relevancy','Text Faithfulness', 'Text Context Relevancy']
    
    results = evaluation_module.evaluate(metrics=METRICS,
                                         query=user_query,
                                         context=context,
                                         image=image,
                                         generated_answer=generated_answer,
                                         reference_answer=reference_answer)
    
    print(results)