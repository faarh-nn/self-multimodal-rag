import os
import time
import pandas as pd
from qa_chain import QAChain
from src.rag_env import MODEL_TYPE, RAG_OUTPUT_DIR, REFERENCE_QA
    
    
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
        context = None
        image = None
        inputs = dict()
        inputs["question"] = user_query
        generated_answer = pipeline.run(inputs)       
        print("GENERATED ANSWER:\n", generated_answer)
        write_to_df(output_df, user_query, reference_answer, generated_answer, context, image, output_file)
    return output_df


if __name__ == "__main__":
    start_time = time.time()

    chain = QAChain(model_type=MODEL_TYPE)

    output_file = os.path.join(RAG_OUTPUT_DIR, f"rag_output_{MODEL_TYPE}_baseline.json")
    df = pd.read_excel(REFERENCE_QA)
    process_dataframe(df, chain, output_file)

    end_time = time.time()
    print("total time %g sec" % (end_time - start_time))
