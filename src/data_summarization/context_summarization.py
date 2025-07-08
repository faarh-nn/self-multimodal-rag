import pandas as pd
import random
import time
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tqdm.auto import tqdm
from typing import List, Tuple
from src.utils.base64_utils.base64_utils import *
from src.rag_env import INPUT_TEXT_DATA, INPUT_IMG_DATA,IMG_SUMMARIES_CACHE_DIR, TEXT_SUMMARIES_CACHE_DIR


class TextSummarizer:
    """  
    A class to summarize texts using OpenAI's models via direct API.
  
    Attributes:
        model_name (str): Name of the OpenAI model to use for summarization.
        cache_path (str): Path to the directory where summaries will be cached as a CSV file.
        model (ChatOpenAI): The OpenAI model instance used for summarization.
        cache_file (str): The complete path to the cached CSV file containing text summaries.
        df (pd.DataFrame): DataFrame to store and manage texts and their corresponding summaries.
    """    
    
    def __init__(self, model_type: str, cache_path: str):
        """  
        Initializes the TextSummarizer object.
  
        :param model_type: The type of model to be used for summarization.
        :param cache_path: The directory path where the summaries will be cached.
        """ 
        self.model_type = model_type

        # Initialize OpenAI model
        self.model = ChatOpenAI(
            model=model_type,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.2,
            max_completion_tokens=500
        )
            
        # Load cached DataFrame if it exists
        self.cache_file = os.path.join(cache_path, f'text_summaries_{self.model_type}.csv')
        if os.path.exists(self.cache_file):
            self.df = pd.read_csv(self.cache_file)
        else:
            # Initialize DataFrame if it doesn't exist
            self.df = pd.DataFrame(columns=['text', 'text_summary'])
            self.df.to_csv(self.cache_file, index=False)
    
    def summarize(self, texts: List[str]) -> List[str]:
        """  
        Generates summaries for a list of texts using the OpenAI model.
  
        :param texts: A list of texts to be summarized.
        :return: A list of summarized texts.
        """  
        print(f"Summarizing texts with OpenAI {self.model_type}")
  
        # Iterate over texts and generate summaries with a progress bar
        with tqdm(total=len(texts), desc="Summarizing texts") as pbar:
            for i, text in enumerate(texts):
                self.df.at[i, 'text'] = text

                pbar.update(1)
  
                # Skip if summary already exists
                if i < len(self.df) and pd.notna(self.df.at[i, 'text_summary']):
                    print(f"Summary for text {i + 1} already exists. Skipping...")
                    continue
  
                # Prompt template
                prompt = f"""Anda adalah asisten yang bertugas merangkum teks untuk proses retrieval.
                Ringkasan ini akan diubah menjadi embedding dan digunakan untuk mengambil atau me-retrieve elemen raw teks sebagai konteks dalam sistem Retrieval Augmented Generation (RAG).
                Buat ringkasan teks yang dioptimalkan dengan baik dan dapat mendukung proses retirval.
                Text: {text}\n"""
  
                try:
                    print(f"Summarizing text {i + 1} of {len(texts)}")
                    summary = self.exponential_backoff_retry(
                        lambda: self.model.invoke([HumanMessage(content=prompt)])
                    )
                    
                    if summary is None:
                        print(f"Failed to generate summary for text {i}")
                        with open('summarization_fails.txt', 'a') as f:
                            f.write(f"Failed to summarize text {i}: Max retries reached\n")
                        continue

                    print(summary.content)
                except Exception as e:
                    print(f"Failed to summarize text {i}: {e}")
                    with open('summarization_fails.txt', 'a') as f:
                        f.write(f"Failed to summarize text {i}: {e}\n")
                    continue
  
                # Update DataFrame with new summary
                self.df.at[i, 'text_summary'] = summary.content
                # Cache the DataFrame after each generation
                self.df.to_csv(self.cache_file, index=False)
  
        return self.df['text_summary'].tolist()

    @staticmethod
    def exponential_backoff_retry(func, *args, **kwargs):
        """
        Fungsi wrapper untuk menangani rate limit dengan exponential backoff.
        Akan mencoba ulang `func()` hingga `max_retries` kali jika terjadi `RateLimitError`.
        """
        max_retries=10
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)  
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

class ImageSummarizer:
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
    
    def summarize(self, image_bytes_list: List[bytes], cache_path: str) -> Tuple[List[str], List[str]]:
        """  
        Generate summaries and base64 encoded strings for images. This function also checks for cached summaries
        to avoid re-processing images. If a summary does not exist, it will generate a new one, update the cache,
        and return the summaries along with their base64 encoded strings.
          
        :param image_bytes_list: A list of image bytes. Each entry in the list should be the binary content of an image file.
        :param cache_path: The file system path where cached summaries are stored. This path is used to store summaries
                           in a CSV file to avoid re-processing images.
        :return: A tuple containing two lists - the first list contains the base64 encoded strings of the images,
                 and the second list contains the textual summaries of the images.
        """
        # Initialize base64 list
        img_base64_list = []

        model_type = "gpt-4o-mini"

        # Load cached DataFrame if it exists
        cache_file = os.path.join(cache_path, f'image_summaries_{model_type}.csv')
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
        else:
            # Initialize DataFrame if it doesn't exist
            df = pd.DataFrame(columns=['image_summary'])
            df.to_csv(cache_file, index=False)

        # Prompt template
        prompt = """Anda adalah asisten yang bertugas meringkas gambar untuk proses retrieval. \n
                    Ringkasan ini akan diubah menjadi embedding dan digunakan untuk mengambil atau me-retrieve elemen raw image sebagai konteks dalam sistem Retrieval Augmented Generation (RAG). \n
                    Untuk itu, buat ringkasan gambar yang komprehensif dan dioptimalkan dengan baik sehingga dapat mendukung proses retrieval. \n
                    Khusus untuk gambar yang mengandung formula matematis, selain melakukan peringkasan, tuliskan juga formula matematis yang ada pada gambar tersebut dalam format latex"""

        # Iterate over image bytes and generate base64 encoded string
        for i, image_bytes in enumerate(image_bytes_list):
            # Convert image bytes to base64
            try:
                img_base64 = encode_image_from_bytes(image_bytes)
                img_base64_list.append(img_base64)
            except Exception as e:
                print(f"Failed to encode img {i}: {str(e)}")
                continue

            # Skip if summary already exists
            if i < len(df):
                print(f"Summary for image {i + 1} already exists. Skipping...")
                continue
            
            print(f"Summarizing image {i + 1} of {len(image_bytes_list)}")

            summary_content = self.exponential_backoff_retry(lambda: self.summarize_image_openai(img_base64, prompt), max_retries=5)

            if summary_content is None:
                print(f"Failed to summarize img {i} after multiple retries.")
                with open('summarization_fails.txt', 'a') as f:
                    f.write(f"Failed to summarize img {i}\n")
                continue  # Lanjutkan ke gambar berikutnya jika tetap gagal

            # Update DataFrame with new summary
            df.at[i, 'image_summary'] = summary_content

            # Cache the DataFrame after each generation
            df.to_csv(cache_file, index=False)

        return img_base64_list, df['image_summary'].tolist()

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
        msg = self.model.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        },
                    ]
                )
            ]
        )
        print(msg.content)
        return msg.content

if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # text summarization
    text_summarizer = TextSummarizer(model_type='gpt-4o-mini', cache_path=TEXT_SUMMARIES_CACHE_DIR)
    df = pd.read_parquet(INPUT_TEXT_DATA)
    texts = list(df.drop_duplicates(subset='text')['text'])
    text_summarizer.summarize(texts)

    # image summarization
    df = pd.read_parquet(INPUT_IMG_DATA)
    # filtered_df = df[df['has_image'] == True]
    images = list(df["image_bytes"])

    model = ChatOpenAI(
            model='gpt-4o-mini',
            api_key=OPENAI_API_KEY,
            temperature=0.2,
            max_completion_tokens=500
        )
    
    image_summarizer = ImageSummarizer(model)
    image_summarizer.summarize(images, cache_path=IMG_SUMMARIES_CACHE_DIR)