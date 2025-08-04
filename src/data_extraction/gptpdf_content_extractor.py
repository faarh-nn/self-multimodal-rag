from gptpdf import parse_pdf
import os
from src.rag_env import MANUALS_DIR, PROCESSED_MANUALS_DIR, PARQUET_DIR
from pypdf import PdfReader
import openai
import time
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import glob
from dotenv import load_dotenv
load_dotenv()


def get_pdf_page_count(pdf_path):
    """Get the number of pages in a PDF file."""    
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        print(f"Failed to get page count for {pdf_path}: {e}")
        return 0


def safe_parse_pdf(*args, retries=5, delay=10, **kwargs):
    """
    Calls parse_pdf with the given arguments, but retries up to the given number of times if
    an OpenAI error occurs (rate limit error or other).

    :param args: Arguments to pass to parse_pdf
    :param retries: Number of times to retry on error
    :param delay: Number of seconds to wait between retries
    :param kwargs: Keyword arguments to pass to parse_pdf
    :return: A tuple of (content, image_paths) if successful, or (None, None) if all retries fail
    """
    for attempt in range(retries):
        try:
            return parse_pdf(*args, **kwargs)
        except openai.error.RateLimitError:
            print(f"Rate limit reached. Retrying in {delay} seconds... (Attempt: {attempt + 1})")
            time.sleep(delay)
            delay *= 2
        except openai.error.OpenAIError as e:
            print(f"OpenAI error: {e}. Retrying... (Attempt: {attempt + 1})")
            time.sleep(delay)
        except Exception as e:
            print(f"General error: {e}")
            break
    return None, None


def run_batch_processing(max_pages_per_batch: int, pause_between_batches: int, input_dir: str, output_root_dir: str, api_key: str = None, prompt: dict = None):
    """
    Process all PDFs in the input directory in batches.

    Given the total number of pages in the current batch exceeds the maximum allowed, process the current batch, wait for a specified amount of time, and reset the batch.

    :param max_pages_per_batch: The maximum number of pages that can be processed in a batch.
    :param pause_between_batches: The number of seconds to wait between processing batches.
    :param output_root_dir: The root directory where output markdown files will be saved.
    """
    all_pdfs = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    current_batch = []
    total_pages = 0

    for filename in all_pdfs:
        pdf_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(output_root_dir, base_name)
        output_md_path = os.path.join(output_dir, f"{base_name}.md")

        if os.path.exists(output_md_path):
            print(f"{filename} already processed. Skipping...")
            continue

        page_count = get_pdf_page_count(pdf_path)

        if total_pages + page_count > max_pages_per_batch and current_batch:
            process_batch(current_batch, output_root_dir, api_key, prompt)
            print(f"Waiting {pause_between_batches} seconds before processing the next batch...")
            time.sleep(pause_between_batches)
            current_batch = []
            total_pages = 0

        current_batch.append((filename, pdf_path))
        total_pages += page_count

    if current_batch:
        process_batch(current_batch, output_root_dir, api_key, prompt,is_final=True)


def process_batch(batch, output_root_dir, api_key=None, prompt=None, is_final=False):
    """
    Process a batch of PDFs by calling safe_parse_pdf on each PDF and moving the output to a new folder.
    
    :param batch: A list of tuples, where each tuple contains the name of the PDF file and its full path.
    :param is_final: Whether this is the final batch. If True, print "terakhir" in the progress message.
    :param output_root_dir: The root directory where output markdown files will be saved.
    """
    label = "Last" if is_final else ""
    total_pages = sum(get_pdf_page_count(path) for _, path in batch)
    print(f"\nProcessing {label}  batch ({total_pages} pages): {[f for f, _ in batch]}")
    for pdf_name, full_path in batch:
        base = os.path.splitext(pdf_name)[0]
        out_dir = os.path.join(output_root_dir, base)
        os.makedirs(out_dir, exist_ok=True)

        content, image_paths = safe_parse_pdf(
            # content dan image_paths tidak perlu di-return karena mereka akan membuat output hasil ekstraksi secara otomatis
            pdf_path=full_path,
            output_dir=out_dir,
            model="gpt-4o",
            api_key=api_key,
            prompt=prompt,
            verbose=False,
        )

        original_md = os.path.join(out_dir, "output.md")
        renamed_md = os.path.join(out_dir, f"{base}.md")
        if os.path.exists(original_md):
            os.rename(original_md, renamed_md)
        else:
            print(f"{original_md} not found.")
    

def create_dataframe_from_pdf(processed_dir: str, output_parquet_dir: str):
        """
        Processes all extracted markdown files and images from PDF processing.
        
        Args:
            processed_dir (str): Directory containing processed PDF output folders
            output_parquet_dir (str, optional): Directory to save parquet files. 
                                            If None, uses processed_dir/parquet
        
        Returns:
            tuple: (text_df_path, image_df_path) - Paths to the saved parquet files
        """
        os.makedirs(output_parquet_dir, exist_ok=True)

        # Process text files
        text_df_path = os.path.join(output_parquet_dir, "extracted_texts.parquet")
        image_df_path = os.path.join(output_parquet_dir, "extracted_images.parquet")

        # Check if files already exist
        if os.path.exists(text_df_path) and os.path.exists(image_df_path):
            print(f"Extracted text and image dataframes already exist at {output_parquet_dir}")
            return text_df_path, image_df_path
        
        # Get all PDF output folders
        pdf_folders = [f for f in os.listdir(processed_dir) 
                    if os.path.isdir(os.path.join(processed_dir, f))]
        
        # Initialize lists to collect data
        text_chunks = []
        image_data = []

        # Initialize text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o-mini",
            chunk_size=1042, # based on mean tokens per page
            chunk_overlap=200, # depends on chunk size
        )

        print(f"Processing {len(pdf_folders)} PDF output folders...")

        # Process each PDF folder
        for pdf_folder in tqdm(pdf_folders):
            folder_path = os.path.join(processed_dir, pdf_folder)
            doc_id = pdf_folder  # Use folder name as doc_id
            
            # Find all markdown files in the folder
            md_files = glob.glob(os.path.join(folder_path, "*.md"))
            
            # Process each markdown file
            for md_file in md_files:
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split text into chunks
                    chunks = text_splitter.split_text(content)
                    
                    # Add each chunk to list
                    for chunk in chunks:
                        text_chunks.append({
                            "doc_id": doc_id,
                            "text": chunk
                        })
                except Exception as e:
                    print(f"Error processing markdown file {md_file}: {e}")
            
            # Find all image files in the folder
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.gif']:
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
            # Process each image file
            for img_file in image_files:
                try:
                    with open(img_file, 'rb') as f:
                        img_bytes = f.read()
                    
                    # Add image data to list
                    image_data.append({
                        "doc_id": doc_id,  # Use PDF name, not image file name
                        "image_bytes": img_bytes
                    })
                except Exception as e:
                    print(f"Error processing image file {img_file}: {e}")
        
        # Create dataframes
        text_df = pd.DataFrame(text_chunks)
        image_df = pd.DataFrame(image_data)
        
        # Save dataframes to parquet
        if len(text_df) > 0:
            text_df.to_parquet(text_df_path, engine='pyarrow', index=False)
            print(f"Saved {len(text_df)} text chunks to {text_df_path}")
        else:
            print("No text chunks found")
        
        if len(image_df) > 0:
            image_df.to_parquet(image_df_path, engine='pyarrow', index=False)
            print(f"Saved {len(image_df)} images to {image_df_path}")
        else:
            print("No images found")
        
        return text_df_path, image_df_path # in case we need the path for further processing

if __name__ == "__main__":
    start_time = time.time()

    prompt = {
        "prompt": "Dengan menggunakan sintaks markdown, ubahlah teks yang dikenali dari gambar ke dalam format markdown. Anda harus memastikan bahwa:\n1. Bahasa output harus sesuai dengan bahasa yang dikenali dalam gambar. Misalnya, jika teks yang dikenali berbahasa Indonesia, maka output juga harus dalam bahasa Indonesia.\n2. Jangan memberikan penjelasan atau menyertakan teks yang tidak relevan; output hanya berupa konten dari gambar. Misalnya, jangan keluarkan frasa seperti 'Berikut adalah teks markdown yang dihasilkan dari konten gambar:'. Sebaliknya, langsung keluarkan output markdown-nya.\n3. Jangan membungkus/enclose konten dengan ```markdown```. Gunakan $$ $$ untuk blok formula, $ $ untuk inline formula, dan abaikan nomor halaman.\n\nSekali lagi, jangan beri penjelasan atau masukkan teks yang tidak berkaitan; keluarkan hanya konten langsung dari gambar.",
        "rect_prompt": "Dalam gambar, beberapa area ditandai dengan kotak merah dan dinamai (%s). Jika suatu area berupa tabel atau gambar, masukkan ke dalam konten output menggunakan format ![](). Jika tidak, keluarkan langsung konten teksnya.",
        "role_prompt": "Anda adalah parser dokumen PDF yang mengekstrak konten gambar menggunakan sintaks markdown dan LaTeX."
    }

    run_batch_processing(
        max_pages_per_batch=50,
        pause_between_batches=60,
        input_dir=MANUALS_DIR,
        output_root_dir=PROCESSED_MANUALS_DIR,
        api_key=os.getenv("OPENAI_API_KEY"),
        prompt=prompt
    )

    text_df_path, image_df_path = create_dataframe_from_pdf(PROCESSED_MANUALS_DIR, PARQUET_DIR)

    end_time = time.time()
    print("total time %g sec" % (end_time - start_time))