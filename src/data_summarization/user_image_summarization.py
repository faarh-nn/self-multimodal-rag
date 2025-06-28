import os
import pandas as pd
import random
import time
import hashlib
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Tuple
from src.utils.base64_utils.base64_utils import *

class UserImageSummarizer:
    """
    A class to summarize images using OpenAI API directly.
    It can encode images in base64, generate textual summaries, and cache these summaries for quick retrieval.
    """
    def __init__(self, model, tokenizer=None):
        """
        Initializes the ImageSummarizer with a specific model and an optional tokenizer.
          
        :param model: The model to be used for generating image summaries. This can be an instance of either
                      AzureChatOpenAI, LlavaNextForConditionalGeneration, or any model that supports image summarization.
        :param tokenizer: The tokenizer to be used with the model, if necessary. This is model-dependent and optional.
        """
        self.model = model
        self.tokenizer=tokenizer
    
    def summarize(self, image_bytes: bytes, cache_path: str) -> str:
        """  
        Generate summaries and base64 encoded strings for images. This function also checks for cached summaries
        to avoid re-processing images. If a summary does not exist, it will generate a new one, update the cache,
        and return the summaries along with their base64 encoded strings.
          
        :param image_bytes: The binary content of an image file.
        :param cache_path: The file system path where cached summaries are stored. This path is used to store summaries
                           in a CSV file to avoid re-processing images.
        :return: A tuple containing two lists - the first list contains the base64 encoded strings of the images,
                 and the second list contains the textual summaries of the images.
        """

        model_type = "gpt-4o-mini"
        cache_file = os.path.join(cache_path, f'image_summaries_{model_type}.csv')

        # Load or create cache file
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
        else:
            # Initialize DataFrame if it doesn't exist
            df = pd.DataFrame(columns=['image_hash', 'image_summary'])
            df.to_csv(cache_file, index=False)

        # Hash dari image untuk identifikasi unik
        img_hash = hashlib.md5(image_bytes).hexdigest()

        # Cek apakah sudah ada dalam cache
        if img_hash in df['image_hash'].values:
            summary = df[df['image_hash'] == img_hash]['image_summary'].values[0]
            print(f"[CACHE] Summary for this image already exists.")
            return summary
        
        # Jika belum ada buat ringkasan
        img_base64 = encode_image_from_bytes(image_bytes)

        # Prompt template
        prompt = """Anda adalah asisten yang bertugas meringkas gambar untuk proses retrieval. \n
                    Ringkasan ini akan diubah menjadi embedding dan digunakan untuk mengambil atau me-retrieve elemen raw teks ataupun image sebagai konteks dalam sistem Retrieval Augmented Generation (RAG).\n
                    Untuk itu, buat ringkasan gambar yang komprehensif dan dioptimalkan dengan baik sehingga dapat mendukung proses retrieval."""

        print("Summarizing new image...")
        summary = self.exponential_backoff_retry(lambda: self.summarize_image_openai(img_base64, prompt), max_retries=5)

        if summary is None:
            raise RuntimeError("Failed to summarize image after multiple retries.")

        # Simpan ke cache
        new_row = pd.DataFrame([{'image_hash': img_hash, 'image_summary': summary}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(cache_file, index=False)

        return summary

    @staticmethod
    def exponential_backoff_retry(func, max_retries=10):
        """
        Fungsi wrapper untuk menangani rate limit dengan exponential backoff.
        Akan mencoba ulang `func()` hingga `max_retries` kali jika terjadi `RateLimitError`.
        """
        for attempt in range(max_retries):
            try:
                return func()  
            except Exception as e:
                error_message = str(e)
                
                # Jika bukan rate limit error, langsung raise
                if "rate limit" not in error_message.lower():
                    raise  

                # Hitung waktu tunggu dengan exponential backoff
                wait_time = min(2 ** attempt + random.uniform(0, 1), 60)
                print(f"Rate limit reached. Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
        
        print("Max retries reached. Giving up.")
        return None
        
    def summarize_image_openai(self, img_base64: str, prompt: str) -> str:
        """
        Summarize an image using OpenAI's vision capabilities through the LangChain interface.
        
        :param img_base64: Base64 encoded image string
        :param prompt: The prompt to guide the image summarization
        :return: The textual summary of the image
        """
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=[
                {"type": "text", "text": "Tolong ringkas gambar berikut ini."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ])
        ]

        response = self.model.invoke(messages)
        print(response.content)
        return response.content