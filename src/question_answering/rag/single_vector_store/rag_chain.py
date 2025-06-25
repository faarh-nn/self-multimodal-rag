
from collections import defaultdict
from langchain_core.documents import Document 
from langchain_core.messages import HumanMessage 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnableLambda, RunnablePassthrough 
from src.utils.base64_utils.base64_utils import *
from src.rag_env import REFERENCE_QA
import pandas as pd
from typing import List
import traceback


class MultimodalRAGChain:
    def __init__(self, model, retriever):
        """
        Multi-modal RAG chain
        The steps are as follows: 
        1. Call the retriever to find relevant documents
        2. Split the retrieved documents into images and texts
        3. Create a QA prompt using the question and retrieved image/text context
        4. Call the LLM and prompt it with the created prompt
        5. Obtain the generated answer from the model
        
        :param model: The model used for answer generation.
        :param tokenizer: The tokenizer used for tokenization.
        :param retriever: The retriever used for text and image retrieval.
        :param df: The Dataframe containing the user questions. The names of the files associated with the
        user questions can be used to filter the document collection for retrieval.
        """
        self.model = model
        self.retriever = retriever
        self.tokenizer = None
        self.df = pd.read_excel(REFERENCE_QA)
        
        self.chain = (
                {
                    "context": self.retriever | RunnableLambda(self.split_image_text_types),
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(self.create_prompt)
                | RunnableLambda(lambda x: self.model.invoke(x))
                | StrOutputParser()
        )
        
        
    def run(self, question: str) -> str:
        """
        Menjalankan RAG chain dengan pertanyaan yang diberikan.
        
        :param question: Pertanyaan pengguna
        :return: Jawaban yang dihasilkan
        """
        try:
            return self.chain.invoke(question)
        except Exception as e:
            traceback.print_exc()
            return f"Error dalam memproses pertanyaan: {str(e)}"
    

    def split_image_text_types(self, docs):
        """
        Split base64-encoded images and texts.
        
        :return: A dictionary with separate entries for texts and base64-encoded images.
        """
        b64_images = []
        texts = []
        for doc in docs:
            # Check if the document is of type Document and extract page_content if so
            if isinstance(doc, Document):
                doc = doc.page_content
            else:
                doc = doc.decode('utf-8')
            if looks_like_base64(doc) and is_image_data(doc):
                doc = resize_base64_image(doc, size=(1300, 600))
                b64_images.append(doc)
            else:
                texts.append(doc)
                
        self.retrieved_docs = defaultdict(list)
        self.retrieved_images = b64_images
        self.retrieved_texts = texts

        return self.retrieved_docs


    def create_prompt(self, data_dict: dict) -> dict:
        """
        Constructs a dictionary containing the model-specific prompt and an image.
        """
        
        qa_prompt = """Anda adalah asisten AI yang ahli menjawab pertanyaan tentang sistem neraca nasional.\n
        Anda akan diberikan beberapa konteks yang terdiri dari teks (termasuk rumus matematis) dan/atau gambar, seperti diagram, gambar tabel, gambar yang berisi rumus matematis dan lainnya.\n
        Gunakan informasi dari teks dan gambar (jika ada) untuk memberikan jawaban atas pertanyaan pengguna.\n
        Hindari ungkapan seperti "menurut teks/gambar yang diberikan" dan sejenisnya—cukup berikan jawaban secara langsung."""
        
        # Gabungkan semua teks menjadi satu string dengan pemisah baris baru
        formatted_texts = "\n".join(data_dict["context"]["texts"]) if data_dict["context"]["texts"] else "No text context available."
        
        # Buat list untuk konten pesan
        message_content = []
        
        # Tambahkan gambar ke pesan jika ada
        if data_dict["context"]["images"]:
            # Untuk OpenAI, kita bisa menambahkan beberapa gambar
            for i, image in enumerate(data_dict["context"]["images"]):
                # Batasi ke 5 gambar saja untuk menghindari token yang terlalu banyak
                if i >= 5:
                    break
                    
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"}
                }
                message_content.append(image_message)
        
        # Tambahkan teks untuk generasi jawaban
        text_message = {
            "type": "text",
            "text": (
                f"{qa_prompt}\n"
                f"Pertanyaan yang diberikan oleh pengguna: {data_dict['question']}\n\n"
                "Text context:\n"
                f"{formatted_texts}"
            )
        }
        message_content.append(text_message)
        
        # Buat pesan dalam format yang sesuai
        return [HumanMessage(content=message_content)]