import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from src.utils.base64_utils.base64_utils import encode_image_from_bytes


class QAChain:
    def __init__(self, model_type, user_images_dir=None):
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
        self.user_images_dir = user_images_dir

        self.model = ChatOpenAI(
            model=model_type,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=400
        )

        self.tokenizer = None
        
        # here we don't have a retriever since we prompt the model directly, without any additional context
        self.chain = (self.prompt_func | self.model | StrOutputParser())

    def encode_image(self, image_path):
        """
        Encode image to base64 string
        """
        try:
            with open(image_path, "rb") as image_file:
                return encode_image_from_bytes(image_file.read())
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            return None
        
    def run(self, inputs):
        return self.chain.invoke(inputs)
    

    def prompt_func(self, data_dict):
        
        qa_prompt = """Anda adalah asisten AI yang ahli menjawab pertanyaan tentang sistem neraca nasional.\n
        Jika ada gambar yang disertakan dalam pertanyaan, analisis gambar tersebut dengan teliti dan gunakan informasi dari gambar untuk menjawab pertanyaan"""
        
        # Get image path if provided
        image_id = data_dict.get('image_id', None)
        
        if image_id and self.user_images_dir:
            # Construct full image path
            image_path = os.path.join(self.user_images_dir, image_id)
            
            # Check if image file exists
            if os.path.exists(image_path):
                # Encode image
                base64_image = self.encode_image(image_path)
                
                if base64_image:
                    # Create multimodal message with image
                    messages = [
                        SystemMessage(content=qa_prompt),
                        HumanMessage(content=[
                            {
                                "type": "text",
                                "text": data_dict['question']
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ])
                    ]
                else:
                    # Fallback to text-only if image encoding fails
                    print(f"Failed to encode image {image_path}, using text-only mode")
                    messages = [
                        SystemMessage(content=qa_prompt),
                        HumanMessage(content=data_dict['question'])
                    ]
            else:
                print(f"Image file not found: {image_path}, using text-only mode")
                messages = [
                    SystemMessage(content=qa_prompt),
                    HumanMessage(content=data_dict['question'])
                ]
        else:
            # Text-only message
            messages = [
                SystemMessage(content=qa_prompt),
                HumanMessage(content=data_dict['question'])
            ]
        
        return messages