import os
import time
import pandas as pd  
from single_vector_store.self_rag_pipeline_summaries import SelfMultimodalRAGPipelineSummaries
from src.rag_env import EMBEDDING_MODEL_TYPE, INPUT_TEXT_DATA, MODEL_TYPE, RAG_OUTPUT_DIR, REFERENCE_QA, VECTORSTORE_PATH_TEXT_ONLY, USER_IMAGES_DIR


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

        relevant_contexts = pipeline.self_rag_chain.get_final_context()
        relevant_texts = relevant_contexts["texts"]
        print("Retrieved texts:", len(relevant_texts))
        context = "\n".join(relevant_texts) if len(relevant_texts) > 0 else []

        write_to_df(output_df, user_query, reference_answer, generated_answer, context, [], user_image_filename, output_file)
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
    
    pipeline = SelfMultimodalRAGPipelineSummaries(model_type=MODEL_TYPE, store_path=VECTORSTORE_PATH_TEXT_ONLY, embedding_model=EMBEDDING_MODEL_TYPE, user_images_dir=USER_IMAGES_DIR)
    texts_df = pipeline.load_text_data(INPUT_TEXT_DATA)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    # the pipeline is indexed only with text, no images are added
    pipeline.index_data(texts=texts, text_filenames=texts_filenames)
    df = pd.read_excel(REFERENCE_QA)

    output_file = os.path.join(RAG_OUTPUT_DIR, f"rag_output_{MODEL_TYPE}_self_rag_text_only.json")
    output_df = process_dataframe(df, pipeline, output_file)

    # Print summary
    if 'user_image_filename' in output_df.columns:
        total_queries = len(output_df)
        queries_with_images = output_df['user_image_filename'].notna().sum()
        queries_without_images = total_queries - queries_with_images
        
        print(f"\nProcessing Summary:")
        print(f"Total queries: {total_queries}")
        print(f"Queries with user images: {queries_with_images}")
        print(f"Queries without user images: {queries_without_images}")

    end_time = time.time()
    print("total time %g sec" % (end_time - start_time))