from langchain_core.messages import HumanMessage
from src.evaluation.evaluators.base_evaluator import BaseEvaluator
from typing import List

"""
Contains evaluator classes for specific metrics with GPT-4o-mini as evaluator model.
The required input arguments vary depending on the metric to be evaluated.
Each Evaluator inherits from the BaseEvaluator and implements the get_prompt method for prompt construction.
Each prompt instructs the model to evaluate the desired metric, it provides a description of the metric,
it provides the required input arguments to the model, and it describes the output format required.
"""


class TextContextRelevancyEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, context: str, model):
        super().__init__(model=model, user_query=user_query, context=context)

    def get_prompt(self, inputs: dict):
        message = {
            "type": "text",
            "text": (
                f"""
            Evaluasi metrik berikut:\n
            text_context_relevancy: Apakah konteks yang diberikan oleh teks "{inputs["context"]}" relevan (mengandung informasi yang berkaitan) terhadap pertanyaan "{inputs["user_query"]}"? (YES atau NO)\n
            Jelaskan langkah demi langkah alasan Anda untuk memastikan bahwa kesimpulan yang diambil benar.
            Berikan alasannya dalam bentuk string, bukan berupa list.
            {self.json_parser.get_format_instructions()}
            """
            ),
        }

        return [HumanMessage(content=[message])]


class ImageContextRelevancyEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, image: List[str], model):
        super().__init__(model=model, user_query=user_query, image=image)
    
    def get_prompt(self, inputs: dict):
        messages = []
        for image in inputs['image']:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image}"},
            }
            messages.append(image_message)

        text_message = {
            "type": "text",
            "text": (
                f"""
            Evaluasi metrik berikut:\n
            image_context_relevancy: Apakah konteks yang diberikan oleh gambar relevan (mengandung informasi yang berkaitan) terhadap pertanyaan "{inputs["user_query"]}"? (YES atau NO)\n
            Jelaskan langkah demi langkah alasan Anda untuk memastikan bahwa kesimpulan Anda benar.
            Berikan alasannya dalam bentuk string, bukan berupa list.
            {self.json_parser.get_format_instructions()}
            """
            ),
        }

        messages.append(text_message)
        return [HumanMessage(content=messages)]


class AnswerRelevancyEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, generated_answer: str, model):
        super().__init__(model=model, user_query=user_query, generated_answer=generated_answer)

    def get_prompt(self, inputs: dict):
        message = {
            "type": "text",
            "text": (
                f"""
            Evaluasi metrik berikut:\n
            answer_relevancy: Apakah jawaban "{inputs["generated_answer"]}" relevan (mengandung informasi yang berkaitan) terhadap pertanyaan "{inputs["user_query"]}"? (YES atau NO)\n
            Jelaskan langkah demi langkah alasan Anda untuk memastikan bahwa kesimpulan Anda benar.
            Berikan alasannya dalam bentuk string, bukan berupa list.
            {self.json_parser.get_format_instructions()}
            """
            ),
        }

        return [HumanMessage(content=[message])]


class AnswerCorrectnessEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, generated_answer: str, reference_answer: str, model):
        super().__init__(model=model, user_query=user_query, reference_answer=reference_answer,
                         generated_answer=generated_answer)

    def get_prompt(self, inputs: dict):
        message = {
            "type": "text",
            "text": (
                f"""
                Anda diberikan sebuah pertanyaan, referensi jawaban yang benar, dan jawaban pelajar. \
                Anda diminta untuk menilai apakah jawaban pelajar benar atau salah berdasarkan referensi jawaban yang ada. \
                Abaikan perbedaan dalam tanda baca (punctuation) dan susunan kata (phrasing) antara jawaban pelajar dan jawaban yang benar. \
                Jawaban pelajar boleh mengandung lebih banyak informasi daripada jawaban yang benar, asalkan tidak mengandung pernyataan yang bertentangan.\
                KUERI PENGGUNA: "{inputs["user_query"]}"\n\
                REFERENSI JAWABAN: "{inputs["reference_answer"]}"\n\
                JAWABAN PELAJAR: "{inputs["generated_answer"]}"\n\
                answer_correctness: Apakah jawaban pelajar benar? (YES atau NO)\n
                Jelaskan langkah demi langkah alasan Anda untuk memastikan bahwa kesimpulan Anda benar.
                Berikan alasannya dalam bentuk string, bukan berupa list.
                {self.json_parser.get_format_instructions()}
                """
            ),
        }

        return [HumanMessage(content=[message])]


class ImageFaithfulnessEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, generated_answer: str, image: List[str], model):
        super().__init__(user_query=user_query, generated_answer=generated_answer, image=image, model=model)

    def get_prompt(self, inputs: dict):
        
        if not inputs["image"]:
            return None

            
        
        messages = []
        for image in inputs['image']:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image}"},
            }
            messages.append(image_message)

        text_message = {
            "type": "text",
            "text": (
                f"""
                Evaluasi metrik berikut:\n
                image_faithfulness: Apakah jawaban sesuai (faithful) dengan konteks yang diberikan oleh gambar, dalam artian apakah jawaban tersebut secara faktual selaras dengan konteks? (YES atau NO)\n
                JAWABAN: "{inputs["generated_answer"]}"\n\
                Jelaskan langkah demi langkah alasan Anda untuk memastikan bahwa kesimpulan Anda benar.
                Berikan alasannya dalam bentuk string, bukan berupa list.
                {self.json_parser.get_format_instructions()}
                """
            ),
        }

        messages.append(text_message)
        return [HumanMessage(content=messages)]


class TextFaithfulnessEvaluator(BaseEvaluator):
    def __init__(self, user_query: str, generated_answer: str, context: str, model):
        super().__init__(user_query=user_query, generated_answer=generated_answer, context=context, model=model)

    def get_prompt(self, inputs: dict):
        
        if not inputs["context"]:
            return None
        
        message = {
            "type": "text",
            "text": (
                f"""
                Evaluasi metrik berikut:\n
                text_faithfulness: Apakah jawaban sesuai (faithful) dengan konteks yang diberikan oleh teks, dalam artian apakah jawaban tersebut secara faktual selaras dengan konteks? (YES atau NO)\n
                JAWABAN: "{inputs["generated_answer"]}"\n\
                TEXT: "{inputs["context"]}"\n\
                Jelaskan langkah demi langkah alasan Anda untuk memastikan bahwa kesimpulan Anda benar.
                Berikan alasannya dalam bentuk string, bukan berupa list.
                {self.json_parser.get_format_instructions()}
                """
            ),
        }

        return [HumanMessage(content=[message])]