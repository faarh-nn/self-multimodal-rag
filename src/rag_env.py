"""
Configuration for the RAG pipeline. Replace the paths with your own actual paths.
The data used here is small sample data to show the expected format, not the real data.
This toy data will not lead to good results.
"""

# model to use for answer synthesis and image summarization.
MODEL_TYPE  = 'gpt-4o-mini'
# text embedding model
EMBEDDING_MODEL_TYPE = "text-embedding-3-small"

# excel file containing questions and reference answers
REFERENCE_QA = r"data/reference_qa.xlsx"

# directory containing input images from user for question answering
USER_IMAGES_DIR = r"data/image_for_question" 

# directory containing the pdf files from which to extract texts and images
MANUALS_DIR = r"knowledge_base"

# directory containing the processed pdf files
PROCESSED_MANUALS_DIR = r"data/processed_manuals"

# directory to save text and image parquet files
PARQUET_DIR = r"data"

# parquet file where extracted texts and image bytes are stored
INPUT_TEXT_DATA = r'data/extracted_texts.parquet' 
INPUT_IMG_DATA = r'data/extracted_images.parquet'

# directory where extracted images are stored
# IMAGES_DIR = r'data/images'

# directories containing csv files with text summaries or image summaries
IMG_SUMMARIES_CACHE_DIR = r"data/image_summaries"
TEXT_SUMMARIES_CACHE_DIR = r"data/text_summaries" 

# directories where vector stores are saved
VECTORSTORE_PATH_SUMMARIES_SINGLE = r"data/vec_and_doc_stores/image_summaries"
VECTORSTORE_PATH_TEXT_ONLY = r"data/vec_and_doc_stores/text_only"

# directory where the output of a RAG pipeline is stored
RAG_OUTPUT_DIR = r"data/rag_outputs"

# directory where the evaluation results for a RAG pipeline are stored
EVAL_RESULTS_PATH = r"data/rag_evaluation_results" # belum kepake