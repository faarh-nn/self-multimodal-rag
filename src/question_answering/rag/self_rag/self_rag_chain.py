import pandas as pd
from langchain_openai import ChatOpenAI
from src.question_answering.rag.self_rag.grade_documents import GradeDocuments
from langchain_core.prompts import ChatPromptTemplate
from src.question_answering.rag.self_rag.grade_hallucinations import GradeHallucinations
from src.question_answering.rag.self_rag.grade_answer import GradeAnswer
from langchain_core.output_parsers import StrOutputParser
from src.utils.base64_utils.base64_utils import *
from langchain.schema import HumanMessage, SystemMessage
from src.question_answering.rag.self_rag.graph_state import GraphState
from langgraph.graph import END, StateGraph, START
import traceback
from langchain_core.runnables import RunnableLambda
import time
import openai


class SelfMultimodalRAGChain:
    def __init__(self, model, retriever):
        """
        Self-RAG enhanced Multimodal RAG chain
        
        :param model: The model used for answer generation.
        :param retriever: The retriever used for text and image retrieval.
        """
        self.model = model
        self.retriever = retriever

        # Initialize graders
        self.init_graders()

        # Build graph workflow
        self.build_workflow()

        # Atribut untuk menyimpan konteks final
        self.retrieved_texts = []
        self.retrieved_images = []
    

    def create_multimodal_hallucination_grader(self):
        """Membuat grader halusinasi multimodal yang bisa melihat teks dan gambar secara visual"""
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )  # Menggunakan model dengan kemampuan vision
        
        structured_llm_hall_grader = llm.with_structured_output(GradeHallucinations)
        
        system = """Anda adalah seorang penilai yang menilai apakah generasi LLM didukung oleh atau didasarkan pada sekumpulan fakta yang diambil.
        Fakta tersebut dapat terdiri dari informasi teks saja, gambar saja, atau teks dan gambar. Periksa semua informasi yang diberikan dengan cermat.
        Jika terdapat gambar, perhatikan konten visualnya dan nilai apakah generasi LLM tersebut berhubungan dengan apa yang ditampilkan.
        Untuk teks, periksa apakah generasi LLM tersebut selaras dengan informasi tekstual yang disediakan.
        Berikan skor biner 'yes' atau 'no'. 'Yes' berarti jawaban didasarkan pada teks dan/atau gambar yang diberikan."""
        
        # Hallucination prompt template yang mendukung multimodal
        def build_multimodal_prompt(args):
            content = []

            # Teks konteks dan generasi
            content.append({
                "type": "text",
                "text": f"Informasi teks:\n\n{args['text_context']}\n\n"
                        f"Generasi LLM untuk dievaluasi:\n\n{args['generation']}\n\n"
            })

            # Gambar-gambar
            for url in args.get("image_urls", []):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{url}"}
                })

            return [
                SystemMessage(content=system),
                HumanMessage(content=content)
            ]
        
        multimodal_chain = RunnableLambda(build_multimodal_prompt) | structured_llm_hall_grader
        return multimodal_chain


    def init_graders(self):
        """Initialize all graders for Self-RAG functionality"""
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"), 
            temperature=0)

        # Document relevance grader
        structured_llm_doc_grader = llm.with_structured_output(GradeDocuments)
        system = """Anda adalah seorang penilai yang menilai relevansi suatu dokumen yang diambil dengan pertanyaan pengguna. \n 
            Proses penilaian ini tidak perlu menjadi pengujian yang ketat, karena tujuannya adalah untuk menyaring dokumen yang keliru atau tidak relevan dengan pertanyaan pengguna. \n
            Jika dokumen berisi kata kunci atau makna semantik yang terkait dengan pertanyaan pengguna, catat dokumen tersebut sebagai dokumen relevan. \n
            Berikan skor biner 'yes' atau 'no' untuk menunjukkan apakah dokumen tersebut relevan dengan pertanyaan."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Dokumen teks yang di-retrieve: \n\n {text} \n\n Pertanyaan pengguna: {question}"),
            ]
        )
        self.text_retrieval_grader = grade_prompt | structured_llm_doc_grader

        # Special image grader with image content
        system_img = """Anda adalah seorang penilai yang menilai relevansi suatu gambar terhadap pertanyaan pengguna. \n
            Perhatikan dengan saksama konten gambar dan tentukan apakah gambar tersebut relevan dengan pertanyaan pengguna atau tidak. \n
            Gambar tersebut mungkin berisi diagram, tabel, rumus, atau informasi visual lainnya. \n
            Berikan skor biner 'yes' atau 'no' untuk menunjukkan apakah gambar tersebut relevan dengan pertanyaan."""
        
        def build_image_prompt(inputs):
            return [
                SystemMessage(content=system_img),
                HumanMessage(content=[
                    {"type": "text", "text": f"User question: {inputs['question']}\n\nPlease assess if this image is relevant:"},
                    {"type": "image_url", "image_url": {"url": inputs["image_url"]}}
                ])
            ]
        
        self.image_retrieval_grader = RunnableLambda(build_image_prompt) | structured_llm_doc_grader

        # Text hallucination grader
        structured_llm_hall_grader = llm.with_structured_output(GradeHallucinations)
        system = """Anda adalah seorang penilai yang menilai apakah generasi LLM didukung oleh atau didasarkan pada sekumpulan fakta yang diambil. \n 
            Berikan skor biner 'yes' atau 'no'. 'Yes' berarti jawaban didukung oleh atau didasarkan pada sekumpulan fakta yang ada."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Kumpulan fakta: \n\n {text} \n\n Generasi LLM: {generation}"),
            ]
        )
        self.text_hallucination_grader = hallucination_prompt | structured_llm_hall_grader
        
        # Multimodal hallucination grader
        self.multimodal_hallucination_grader = self.create_multimodal_hallucination_grader()

        # Answer grader
        structured_llm_ans_grader = llm.with_structured_output(GradeAnswer)
        system = """Anda adalah seorang penilai yang menilai apakah sebuah jawaban menjawab / menyelesaikan suatu pertanyaan. \n 
            Berikan skor biner 'yes' atau 'no'. 'Yes' berarti bahwa jawaban yang ada menjawab / menyelesaikan pertanyaan tersebut."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Pertanyaan pengguna: \n\n {question} \n\n Generasi LLM: {generation}"),
            ]
        )
        self.answer_grader = answer_prompt | structured_llm_ans_grader

        # Question rewriter
        system = """Anda adalah seorang penulis ulang pertanyaan yang mengubah input pertanyaan ke versi yang lebih baik dan dioptimalkan untuk proses retrieval di vectorstore. \n
        Perhatikan input pertanyaan yang ada dan coba pahami maksud atau makna semantik yang mendasarinya."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Berikut adalah input pertanyaannya: \n\n {question} \n Formulasi ulang dan optimalkan pertanyaan tersebut.",
                ),
            ]
        )
        self.question_rewriter = re_write_prompt | llm | StrOutputParser()

    
    def safe_invoke(self, model_callable, input_data, max_retries=10):
        """
        Safely invoke a model with exponential backoff in case of rate limits.

        :param model_callable: A callable model (e.g., self.model.invoke, self.text_hallucination_grader.invoke)
        :param input_data: Input dictionary to pass to the model
        :param max_retries: Number of retry attempts
        :return: Model response
        """
        for attempt in range(max_retries):
            try:
                return model_callable(input_data)
            except openai.RateLimitError:
                wait_time = 2 ** attempt
                print(f"[RateLimitError] Attempt {attempt + 1}: waiting {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"[Unexpectederror] {e}")
                raise
        raise Exception("Max retries exceeded due to rate limit.")

    
    def retrieve(self, state):
        """Retrieve documents"""
        print("---RETRIEVE---")
        question = state["question"]
        
        # Retrieval
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}
    

    def split_image_text_types(self, state):
        """Split retrieved documents into images and texts"""
        print("---SPLIT DOCUMENTS---")
        docs = state["documents"]
        
        b64_images = []
        texts = []
        for doc in docs:
            # Check if the document is of type Document and extract page_content if so
            if hasattr(doc, 'page_content'):
                doc_content = doc.page_content
            else:
                doc_content = doc.decode('utf-8') if isinstance(doc, bytes) else doc
                
            if looks_like_base64(doc_content) and is_image_data(doc_content):
                b64_images.append(doc_content)
            else:
                texts.append(doc_content)
        
        return {
            "question": state["question"], 
            "texts": texts,
            "image_urls": b64_images
        }
    

    def grade_documents(self, state):
        """Determines whether the retrieved documents are relevant to the question."""
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        texts = state.get("texts", [])
        image_urls = state.get("image_urls", [])
        
        filtered_image_urls = []
        filtered_texts = []
        
        # Process text documents
        for text in texts:
            score = self.safe_invoke(
                self.text_retrieval_grader.invoke,
                {"question": question, "text": text}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: TEXT DOCUMENT RELEVANT---")
                filtered_texts.append(text)
            else:
                print("---GRADE: TEXT DOCUMENT NOT RELEVANT---")
        
        # Process images with visual evaluation through 
        print(f"Processing {len(image_urls)} images for relevance grading")
        if image_urls:
            for url in image_urls:
                try:
                    # Grade image using image_url for visual assessment
                    score = self.safe_invoke(
                        self.image_retrieval_grader.invoke,
                        {"question": question, "image_url": f"data:image/png;base64,{url}"}
                    )
                    grade = score.binary_score
                    if grade == "yes":
                        print("---GRADE: IMAGE RELEVANT---")
                        # filtered_docs.append(img_doc)
                        filtered_image_urls.append(url)
                        # Get corresponding b64 image
                        # img_b64 = state["images"][i] if i < len(state["images"]) else None
                        # if img_b64:
                        #     filtered_images.append(img_b64)
                    else:
                        print("---GRADE: IMAGE NOT RELEVANT---")
                except Exception as e:
                    print(f"Error grading image: {str(e)}")
        
        return {
            "question": question,
            "texts": filtered_texts,
            "image_urls": filtered_image_urls
        }
    

    def transform_query(self, state):
        """Transform the query to produce a better question."""
        print("---TRANSFORM QUERY---")
        question = state["question"]
        
        # Re-write question
        better_question = self.safe_invoke(
            self.question_rewriter.invoke,
            {"question": question}
        )
        return {
            "question": better_question
        }
    

    def generate(self, state):
        """Generate answer"""
        print("---GENERATE---")
        question = state["question"]
        texts = state.get("texts", [])
        image_urls = state.get("image_urls", [])

        # Simpan konteks final yang akan digunakan sebagai atribut class
        self.final_texts = texts
        self.final_image_urls = image_urls
        
        # Create context dict for prompt creation
        # context = {"image_urls": image_urls, "texts": texts}
        context = {"image_urls": image_urls, "texts": texts}
        
        # Create prompt
        prompt = self.create_prompt({"context": context, "question": question})
        
        # Generate answer
        generation = self.safe_invoke(
            self.model.invoke,
            prompt
        )
        
        if isinstance(generation, str):
            generation_text = generation
        else:
            # Handle ChatMessage object or other response types
            generation_text = generation.content if hasattr(generation, 'content') else str(generation)
            
        return {
            "question": question, 
            "generation": generation_text,
            "texts": texts,
            "image_urls": image_urls
        }
    

    def decide_to_generate(self, state):
        """Determines whether to generate an answer, or re-generate a question."""
        print("---ASSESS GRADED DOCUMENTS---")
        texts = state.get("texts", [])
        image_urls = state.get("image_urls", [])
        
        if not texts and not image_urls:
            # All documents have been filtered
            # We will re-generate a new query
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"
    

    def grade_generation_v_documents_and_question(self, state):
        """Determines whether the generation is grounded in the document and answers question."""
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        texts = state.get("texts", [])
        image_urls = state.get("image_urls", [])
        generation = state["generation"]
        
        # Gabungkan semua teks menjadi satu string
        text_context = "\n\n".join(texts) if texts else "No text context available."

        # Batasi jumlah gambar untuk menghindari overflow token
        limited_image_urls = image_urls[:5] if image_urls else []
        
        # Jika memiliki gambar, gunakan grader multimodal
        if limited_image_urls:
            score = self.safe_invoke(
                self.multimodal_hallucination_grader.invoke,
                {
                    "text_context": text_context,
                    "generation": generation,
                    "image_urls": limited_image_urls
                }
            )
        else:
            # Jika hanya teks, gunakan grader halusinasi regular
            score = self.safe_invoke(
                self.text_hallucination_grader.invoke,
                {
                    "text": text_context,
                    "generation": generation
                }
            )
        
        grade = score.binary_score
        
        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.safe_invoke(
                self.answer_grader.invoke,
                {"question": question, "generation": generation}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    

    def create_prompt(self, data_dict: dict) -> list:
        """
        Constructs a prompt containing the model-specific prompt and an image.
        """
        
        qa_prompt = """Anda adalah asisten AI yang ahli menjawab pertanyaan tentang sistem neraca nasional.\n
        Anda akan diberikan beberapa konteks yang terdiri dari teks (termasuk rumus matematis) dan/atau gambar, seperti diagram, gambar tabel, gambar yang berisi rumus matematis dan lainnya.\n
        Gunakan informasi dari teks dan gambar (jika ada) untuk memberikan jawaban atas pertanyaan pengguna.\n
        Hindari ungkapan seperti "menurut teks/gambar yang diberikan" dan sejenisnya, cukup berikan jawaban secara langsung."""
        
        # Gabungkan semua teks menjadi satu string dengan pemisah baris baru
        formatted_texts = "\n".join(data_dict["context"]["texts"]) if data_dict["context"]["texts"] else "No text context available."
        
        # Buat list untuk konten pesan
        message_content = []
        
        # Tambahkan gambar ke pesan jika ada
        if data_dict["context"]["image_urls"]:
            # Untuk OpenAI, kita bisa menambahkan beberapa gambar
            for i, url in enumerate(data_dict["context"]["image_urls"]):
                # Batasi ke 5 gambar saja untuk menghindari token yang terlalu banyak
                if i >= 5:
                    break
                    
                image_message = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{url}",
                        "detail": "auto"
                    }
                }
                message_content.append(image_message)
        
        # Tambahkan teks untuk generasi jawaban
        text_message = {
            "type": "text",
            "text": (
                f"{qa_prompt}\n"
                f"Pertanyaan yang diberikan oleh pengguna: {data_dict['question']}\n\n"
                "Text context:\n"
                f"{formatted_texts}"
            )
        }
        message_content.append(text_message)
        
        # Buat pesan dalam format yang sesuai
        return [HumanMessage(content=message_content)]
    

    def get_final_context(self):
        """
        Mengembalikan konteks final yang digunakan LLM untuk menghasilkan jawaban.
        
        :return: Dictionary berisi image_urls dan texts final
        """
        return {
            "image_urls": self.final_image_urls,
            "texts": self.final_texts
        }
    

    def build_workflow(self):
        """Build the workflow graph for Self-RAG"""
        workflow = StateGraph(GraphState)
        
        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("split_documents", self.split_image_text_types)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        
        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "split_documents")
        workflow.add_edge("split_documents", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        
        # Compile
        self.app = workflow.compile()
    

    def run(self, question: str) -> str:
        """
        Run the Self-Multimodal RAG chain with a given question.
        
        :param question: User question
        :return: The generated answer
        """
        try:
            inputs = {"question": question}
            result = None
            
            # Run the graph and collect the final output
            for output in self.app.stream(inputs, {"recursion_limit": 50}):
                result = output
                
            # Return the final generation
            if result:
                for key, value in result.items():
                    if "generation" in value:
                        return value["generation"]
            
            return "Maaf, saya belum memiliki cukup informasi untuk menjawab pertanyaan ini. Anda boleh mencoba mengajukan pertanyaan lain."
            
        except Exception as e:
            if "Recursion limit" in str(e):
                return "Mohon maaf, saya belum memiliki informasi yang memadai untuk menjawab pertanyaan tersebut secara akurat. Silakan ajukan pertanyaan lain yang relevan dengan topik neraca nasional, mulai dari konsep, definisi, metodologi, hingga analisis. Saya akan dengan senang hati membantu Anda."
            
            traceback.print_exc()
            return f"Error dalam memproses pertanyaan: {str(e)}"