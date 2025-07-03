import logging
import os
import pandas as pd
from typing import List, Tuple, Optional
from src.data_summarization.context_summarization import ImageSummarizer, TextSummarizer
from src.data_summarization.user_image_summarization import UserImageSummarizer
from langchain_openai import ChatOpenAI
from src.question_answering.rag.self_rag.self_rag_chain import SelfMultimodalRAGChain
from src.question_answering.rag.single_vector_store.retrieval import SummaryStoreAndRetriever
from src.rag_env import EMBEDDING_MODEL_TYPE, IMG_SUMMARIES_CACHE_DIR, INPUT_TEXT_DATA, INPUT_IMG_DATA,MODEL_TYPE, TEXT_SUMMARIES_CACHE_DIR, VECTORSTORE_PATH_SUMMARIES_SINGLE, USER_IMAGES_DIR


class SelfMultimodalRAGPipelineSummaries:
    """
    Initializes the Multimodal RAG pipeline.
    Answers a user query retrieving additional context from texts and images within a document collection.
    Can be used with different models for answer generation (AzureOpenAI and LLaVA).
    Transforms images into textual summaries and embeds the image summaries, the texts, and the query into a single vector store.
    
    Attributes:
        model_type (str): Type of the model to use for answer synthesis.
        store_path (str): Path to the directory where the vector database is stored.
        embedding_model (str): Text embedding model used to embed texts and image summaries.
        model: The model used for answer synthesis loaded based on `model_type`.
        tokenizer: The tokenizer used for tokenization. Can be None.
        text_summarizer (TextSummarizer): Can be used to summarize texts before retrieving them.
        image_summarizer: (ImageSummarizer): Used to generate textual summaries from images.
        sotre_and_retriever (SummaryStoreAndRetriever): Retrieval using CLIP embeddings for images.
        rag_chain (MultimodalRAGChain): RAG chain performing the QA task.
    """
    def __init__(self, model_type, store_path, embedding_model, user_images_dir=None):
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        print(f"Using OpenAI model '{model_type}' for answer generation")
        self.model = ChatOpenAI(
            model=model_type,
            api_key=OPENAI_API_KEY,
            temperature=0.2,
            max_tokens=400 
        )

        self.text_summarizer = TextSummarizer(model_type=model_type, cache_path=TEXT_SUMMARIES_CACHE_DIR)
        self.image_summarizer = ImageSummarizer(self.model)
        self.user_image_summarizer = UserImageSummarizer(self.model)
        self.store_and_retriever = SummaryStoreAndRetriever(embedding_model=embedding_model,
                                                            store_path=f"{store_path}_{model_type}")
        self.self_rag_chain = SelfMultimodalRAGChain(model=self.model, retriever=self.store_and_retriever.retriever)

        self.user_images_dir = user_images_dir

    def load_text_data(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_parquet(path)
        # drop duplicate entries in the texts column of the dataframe
        texts = df.drop_duplicates(subset='text')[["text", "doc_id"]]
        return texts    

    def load_img_data(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_parquet(path)
        img = df[["doc_id", "image_bytes"]]
        images = img.dropna(subset=['image_bytes'])  
        return images

    def load_user_image(self, image_filename: str) -> Optional[bytes]:
        """
        Load user input image from file.
        
        Args:
            image_filename (str): Nama file gambar yang akan dimuat
            
        Returns:
            Optional[bytes]: Image bytes jika berhasil dimuat, None jika gagal
        """
        if not image_filename:
            return None
            
        image_path = os.path.join(self.user_images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} not found")
            return None
            
        try:
            with open(image_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None

    def summarize_user_image(self, image_filename: str) -> Optional[str]:
        """
        Meringkas gambar input dari user.
        
        Args:
            image_filename (str): Nama file gambar yang akan diringkas
            
        Returns:
            Optional[str]: Ringkasan gambar dalam bentuk teks, None jika gagal
        """
        if not image_filename:
            return None
        
        # Load image bytes
        image_bytes = self.load_user_image(image_filename)
        if image_bytes is None:
            return None
            
        try:
            print(f"Summarizing user input image: {image_filename}")
            
            # Gunakan image summarizer untuk meringkas gambar
            image_summaries = self.user_image_summarizer.summarize(
                image_bytes, 
                cache_path=f"{IMG_SUMMARIES_CACHE_DIR}/user_input"
            )
            
            if image_summaries:
                return image_summaries
            else:
                print(f"Failed to generate summary for image: {image_filename}")
                return None
                
        except Exception as e:
            print(f"Error summarizing user image {image_filename}: {str(e)}")
            return None

    def enhance_query_with_image(self, question: str, image_filename: Optional[str] = None) -> str:
        """
        Menggabungkan pertanyaan user dengan ringkasan gambar jika ada.
        
        Args:
            question (str): Pertanyaan original dari user
            image_filename (Optional[str]): Nama file gambar yang menyertai pertanyaan
            
        Returns:
            str: Enhanced query yang menggabungkan pertanyaan dan ringkasan gambar
        """
        if not image_filename:
            return question
            
        image_summary = self.summarize_user_image(image_filename)
        
        if image_summary:
            enhanced_query = f"""Pertanyaan: {question}\n
            Informasi gambar yang disertakan: {image_summary}\n\n
            Berdasarkan pertanyaan dan konteks gambar di atas, berikan jawaban yang komprehensif."""
            
            print(f"Enhanced query created with image context from: {image_filename}")
            return enhanced_query
        else:
            print(f"Failed to enhance query with image: {image_filename}. Using original question.")
            return question

    def summarize_data(self, texts: List[str], images: List[bytes], cache_file:str) -> Tuple[List[str], List[str], List[str]]:
        # Summarize images
        img_base64_list, image_summaries = self.image_summarizer.summarize(images, cache_file)
        # Summarize texts
        if texts:
            text_summaries = self.text_summarizer.summarize(texts)
            return text_summaries, image_summaries, img_base64_list
        return texts, image_summaries, img_base64_list

    def index_data(self, texts: List[str]=None, text_summaries: List[str]=None,
                   image_summaries: List[str]=None, images_base64: List[str]=None,
                   text_filenames: List[str]=None, image_filenames: List[str]=None):
        if text_summaries:
            print("Adding text summaries to store")
            self.store_and_retriever.add_docs(text_summaries, texts, text_filenames)
        if texts:
            print("Adding texts to store")
            self.store_and_retriever.add_docs(texts, texts, text_filenames)
        if image_summaries:
            print("Adding image summaries to store")
            self.store_and_retriever.add_docs(image_summaries, images_base64, image_filenames)

    def answer_question(self, question: str, image_filename: Optional[str] = None) -> str:
        """
        Menjawab pertanyaan user dengan mempertimbangkan gambar input jika ada.
        
        Args:
            question (str): Pertanyaan dari user
            image_filename (Optional[str]): Nama file gambar yang menyertai pertanyaan
            
        Returns:
            str: Jawaban dari sistem RAG
        """
        enhanced_query = self.enhance_query_with_image(question, image_filename)
        return self.self_rag_chain.run(enhanced_query)

def main():
    pipeline = SelfMultimodalRAGPipelineSummaries(
        model_type=MODEL_TYPE,
        store_path=VECTORSTORE_PATH_SUMMARIES_SINGLE,
        embedding_model=EMBEDDING_MODEL_TYPE,
        user_images_dir=USER_IMAGES_DIR   
    )
    
    texts_df = pipeline.load_data(INPUT_TEXT_DATA)
    images_df = pipeline.load_data(INPUT_IMG_DATA)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["doc_id"]]["doc_id"].tolist()
    img_base64_list, image_summaries = pipeline.image_summarizer.summarize(images, IMG_SUMMARIES_CACHE_DIR)

    pipeline.index_data(texts, image_summaries=image_summaries, 
                        images_base64=img_base64_list, 
                        image_filenames=image_filenames, 
                        text_filenames=texts_filenames)
    
    # Contoh dengan gambar
    question_with_image = "I want to change the behaviour of the stations to continue, if a moderate error occurrs. How can I do this?"
    answer_with_image = pipeline.answer_question(question_with_image, "example_image.jpg")
    
    # Contoh tanpa gambar
    question_without_image = "What is the default behavior when an error occurs?"
    answer_without_image = pipeline.answer_question(question_without_image)
    
    print("Answer with image context:", answer_with_image)
    print("Answer without image:", answer_without_image)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("An error occurred during execution", exc_info=True)
