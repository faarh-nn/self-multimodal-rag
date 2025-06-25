import os
import time  
import pandas as pd
from single_vector_store.self_rag_pipeline_summaries import SelfMultimodalRAGPipelineSummaries
from src.rag_env import *


def write_to_df(df, user_query, reference_answer, generated_answer, context, image, output_file):
    df.loc[len(df)] = [user_query, reference_answer, generated_answer, context, image]
    df.to_json(output_file, orient="records", indent=2)


def process_dataframe(input_df, pipeline, output_file, output_df=None):
    if not output_df:
        columns = ['user_query', 'reference_answer', 'generated_answer', 'context', 'image']
        output_df= pd.DataFrame(columns=columns)
    for index, row in input_df.iterrows():
        print(f"Processing query no. {index+1}...")
        user_query = input_df["question"][index]
        print("USER QUERY:\n", user_query)
        reference_answer = input_df["reference_answer"][index]
        print("REFERENCE ANSWER:", reference_answer)
        generated_answer = pipeline.answer_question(user_query)
        print("GENERATED ANSWER:\n", generated_answer)
        relevant_text_and_images = pipeline.self_rag_chain.get_final_context()
        relevant_images = relevant_text_and_images["image_urls"]
        relevant_texts = relevant_text_and_images["texts"]
        print("Retrieved images:", len(relevant_images), ", Retrieved texts:", len(relevant_texts))
        context = "\n".join(relevant_texts) if len(relevant_texts) > 0 else []
        image = relevant_images if len(relevant_images) > 0 else []
        write_to_df(output_df, user_query, reference_answer, generated_answer, context, image, output_file)
    return output_df    
    
def run_pipeline_with_summaries_single(qa_model, embedding_model, vectorstore_path, input_text_df, input_img_df, reference_qa, output_dir, img_summaries_dir):
    summaries_pipeline = SelfMultimodalRAGPipelineSummaries(model_type=qa_model,
                                                        store_path=vectorstore_path,
                                                        embedding_model=embedding_model,
                                                        reference_qa_path=reference_qa)
    
    texts_df = summaries_pipeline.load_text_data(input_text_df)
    images_df = summaries_pipeline.load_img_data(input_img_df)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    images, image_filenames = images_df[["image_bytes"]]["image_bytes"].tolist(), images_df[["doc_id"]]["doc_id"].tolist()
    img_base64_list, image_summaries = summaries_pipeline.image_summarizer.summarize(images, img_summaries_dir)
    summaries_pipeline.index_data(image_summaries=image_summaries, 
                                  images_base64=img_base64_list, image_filenames=image_filenames, 
                                  texts = texts, text_filenames=texts_filenames)
    
    df = pd.read_excel(reference_qa)
    
    output_file = os.path.join(output_dir, f"rag_output_{qa_model}_self_multimodal_rag_summaries_single.json")
    output_df = process_dataframe(df, summaries_pipeline, output_file)
    return output_df   
  
if __name__ == "__main__":
    start_time = time.time()

    # uncomment one of the following options to run multimodal RAG either with CLIP embedings or with image summaries
    # and either with a single vector store for both modalities or a dedicated one for each modality.
    rag_results_summaries_single = run_pipeline_with_summaries_single(qa_model=MODEL_TYPE,
                                                                      vectorstore_path=VECTORSTORE_PATH_SUMMARIES_SINGLE,
                                                                      embedding_model=EMBEDDING_MODEL_TYPE, input_text_df=INPUT_TEXT_DATA,
                                                                      input_img_df=INPUT_IMG_DATA,
                                                                      reference_qa=REFERENCE_QA, output_dir=RAG_OUTPUT_DIR,
                                                                      img_summaries_dir=IMG_SUMMARIES_CACHE_DIR)
    
    end_time = time.time()
    print("total time %g sec" % (end_time - start_time))