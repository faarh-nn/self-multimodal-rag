import os
import time  
import pandas as pd
from single_vector_store.rag_pipeline_summaries import MultimodalRAGPipelineSummaries
from src.rag_env import *


def write_to_df(df, user_query, reference_answer, generated_answer, context, image, user_image_filename, output_file):
    df.loc[len(df)] = [user_query, reference_answer, generated_answer, context, image, user_image_filename]
    df.to_json(output_file, orient="records", indent=2)


def process_dataframe(input_df, pipeline, output_file, output_df=None):
    if not output_df:
        columns = ['user_query', 'reference_answer', 'generated_answer', 'context', 'image', 'user_image_filename']
        output_df= pd.DataFrame(columns=columns)

    for index, row in input_df.iterrows():
        print(f"Processing query no. {index+1}...")

        user_query = input_df["question"][index]
        print("USER QUERY:\n", user_query)

        reference_answer = input_df["reference_answer"][index]
        print("REFERENCE ANSWER:", reference_answer)

        # Ambil nama file gambar jika ada
        user_image_filename = None
        if "gambar_id" in input_df.columns and pd.notna(input_df["gambar_id"][index]):
            user_image_filename = input_df["gambar_id"][index]
            print(f"USER IMAGE: {user_image_filename}")
        else:
            print("USER IMAGE: None")

        generated_answer = pipeline.answer_question(user_query, user_image_filename)
        print("GENERATED ANSWER:\n", generated_answer)

        relevant_images = pipeline.rag_chain.retrieved_images
        relevant_texts = pipeline.rag_chain.retrieved_texts
        print("Retrieved images:", len(relevant_images), ", Retrieved texts:", len(relevant_texts))
        context = "\n".join(relevant_texts) if len(relevant_texts) > 0 else []
        image = relevant_images if len(relevant_images) > 0 else []

        write_to_df(output_df, user_query, reference_answer, generated_answer, context, image, user_image_filename, output_file)
    return output_df    
    
def run_pipeline_with_summaries_single(qa_model, embedding_model, vectorstore_path, input_text_df, input_img_df, reference_qa, output_dir, img_summaries_dir, user_images_dir=None):
    summaries_pipeline = MultimodalRAGPipelineSummaries(
        model_type=qa_model,
        store_path=vectorstore_path,
        embedding_model=embedding_model,
        user_images_dir=user_images_dir
    )
    
    texts_df= summaries_pipeline.load_text_data(input_text_df)
    images_df = summaries_pipeline.load_img_data(input_img_df)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["doc_id"]]["doc_id"].tolist()
    img_base64_list, image_summaries = summaries_pipeline.image_summarizer.summarize(images, img_summaries_dir)
    summaries_pipeline.index_data(image_summaries=image_summaries, 
                                  images_base64=img_base64_list, image_filenames=image_filenames, 
                                  texts = texts, text_filenames=texts_filenames)
    
    df = pd.read_excel(reference_qa)

    # Validasi kolom yang diperlukan
    required_columns = ['question', 'reference_answer']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in reference QA file")
    
    # Kolom gambar_id opsional
    if 'gambar_id' not in df.columns:
        print("Warning: 'gambar_id' column not found. All queries will be processed without user images.")
        df['gambar_id'] = None
    
    output_file = os.path.join(output_dir, f"rag_output_{qa_model}_multimodal_rag.json")
    output_df = process_dataframe(df, summaries_pipeline, output_file)
    return output_df

def validate_user_images_directory(user_images_dir, reference_qa_file):
    """
    Validasi bahwa semua gambar yang direferensikan dalam dataset tersedia.
    
    Args:
        user_images_dir: Directory gambar user
        reference_qa_file: Path ke file Q&A referensi
    """
    if not user_images_dir or not os.path.exists(user_images_dir):
        print(f"Warning: User images directory '{user_images_dir}' not found or not specified")
        return
    
    # Load dataset
    if reference_qa_file.endswith('.xlsx'):
        df = pd.read_excel(reference_qa_file)
    elif reference_qa_file.endswith('.csv'):
        df = pd.read_csv(reference_qa_file)
    else:
        return
    
    if 'gambar_id' not in df.columns:
        return
    
    # Cek gambar yang ada
    missing_images = []
    for idx, row in df.iterrows():
        if pd.notna(row['gambar_id']):
            image_path = os.path.join(user_images_dir, row['gambar_id'])
            if not os.path.exists(image_path):
                missing_images.append(row['gambar_id'])
    
    if missing_images:
        print(f"Warning: {len(missing_images)} referenced images not found:")
        for img in missing_images[:5]:  # Show first 5 missing images
            print(f"  - {img}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")
    else:
        print("All referenced user images found successfully")   
  
if __name__ == "__main__":  
    start_time = time.time()

    validate_user_images_directory(USER_IMAGES_DIR, REFERENCE_QA)

    # uncomment one of the following options to run multimodal RAG either with CLIP embedings or with image summaries
    # and either with a single vector store for both modalities or a dedicated one for each modality.
    rag_results_summaries_single = run_pipeline_with_summaries_single(qa_model=MODEL_TYPE,
                                                                      vectorstore_path=VECTORSTORE_PATH_SUMMARIES_SINGLE,
                                                                      embedding_model=EMBEDDING_MODEL_TYPE, input_text_df=INPUT_TEXT_DATA,
                                                                      input_img_df=INPUT_IMG_DATA,
                                                                      reference_qa=REFERENCE_QA, output_dir=RAG_OUTPUT_DIR,
                                                                      img_summaries_dir=IMG_SUMMARIES_CACHE_DIR,
                                                                      user_images_dir=USER_IMAGES_DIR)
    
    # Print summary
    if 'user_image_filename' in rag_results_summaries_single.columns:
        total_queries = len(rag_results_summaries_single)
        queries_with_images = rag_results_summaries_single['user_image_filename'].notna().sum()
        queries_without_images = total_queries - queries_with_images
        
        print(f"\nProcessing Summary:")
        print(f"Total queries: {total_queries}")
        print(f"Queries with user images: {queries_with_images}")
        print(f"Queries without user images: {queries_without_images}")

    end_time = time.time()
    print("total time %g sec" % (end_time - start_time))