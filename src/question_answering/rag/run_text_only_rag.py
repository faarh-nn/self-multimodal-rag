import os
import time
import pandas as pd  
from single_vector_store.rag_pipeline_summaries import MultimodalRAGPipelineSummaries
from src.rag_env import EMBEDDING_MODEL_TYPE, INPUT_TEXT_DATA, MODEL_TYPE, RAG_OUTPUT_DIR, REFERENCE_QA, VECTORSTORE_PATH_TEXT_ONLY


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
        relevant_texts = pipeline.rag_chain.retrieved_texts
        print("Retrieved texts:", len(relevant_texts))
        context = "\n".join(relevant_texts) if len(relevant_texts) > 0 else []
        write_to_df(output_df, user_query, reference_answer, generated_answer, context, [], output_file)
    return output_df


  
if __name__ == "__main__":
    start_time = time.time()

    pipeline = MultimodalRAGPipelineSummaries(MODEL_TYPE, VECTORSTORE_PATH_TEXT_ONLY, EMBEDDING_MODEL_TYPE)
    texts_df = pipeline.load_text_data(INPUT_TEXT_DATA)
    texts, texts_filenames = texts_df[["text"]]["text"].tolist(), texts_df[["doc_id"]]["doc_id"].tolist()
    # the pipeline is indexed only with text, no images are added
    pipeline.index_data(texts=texts, text_filenames=texts_filenames)
    df = pd.read_excel(REFERENCE_QA)

    output_file = os.path.join(RAG_OUTPUT_DIR, f"rag_output_{MODEL_TYPE}_rag_text_only.json")
    output_df = process_dataframe(df, pipeline, output_file)

    end_time = time.time()
    print("total time %g sec" % (end_time - start_time))