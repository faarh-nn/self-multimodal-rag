import os
import time
import pandas as pd
from qa_chain import QAChain
from src.rag_env import MODEL_TYPE, RAG_OUTPUT_DIR, REFERENCE_QA, USER_IMAGES_DIR
    
    
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

        # Get image_id from dataset
        image_id = input_df.get("gambar_id", pd.Series([None] * len(input_df)))[index]
        if pd.isna(image_id) or image_id == "":
            image_id = None

        context = None
        image = []
        user_image_filename = image_id

        inputs = dict()
        inputs["question"] = user_query
        inputs["image_id"] = image_id

        try:
            generated_answer = pipeline.run(inputs)       
            print("GENERATED ANSWER:\n", generated_answer)
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            generated_answer = f"Error: {str(e)}" 
              
        write_to_df(output_df, user_query, reference_answer, generated_answer, context, image, user_image_filename, output_file)
    return output_df


if __name__ == "__main__":
    start_time = time.time()

    chain = QAChain(model_type=MODEL_TYPE, user_images_dir=USER_IMAGES_DIR)

    output_file = os.path.join(RAG_OUTPUT_DIR, f"rag_output_{MODEL_TYPE}_baseline.json")
    df = pd.read_excel(REFERENCE_QA)

    baseline_result = process_dataframe(df, chain, output_file)

    # Print summary
    if 'user_image_filename' in baseline_result.columns:
        total_queries = len(baseline_result)
        queries_with_images = baseline_result['user_image_filename'].notna().sum()
        queries_without_images = total_queries - queries_with_images
        
        print(f"\nProcessing Summary:")
        print(f"Total queries: {total_queries}")
        print(f"Queries with user images: {queries_with_images}")
        print(f"Queries without user images: {queries_without_images}")

    end_time = time.time()
    print("total time %g sec" % (end_time - start_time))
