import os
import uuid
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import MultiVectorRetriever 
from langchain_chroma import Chroma 
from langchain_core.documents import Document
from langchain.storage import LocalFileStore
from typing import List


class SummaryStoreAndRetriever:
    """  
    A class providing a document store and a vector store to contain texts and images and their embeddings.
    The class also provides a retriever to find documents that are relevant for a query.
    Retrieval is performed using the embeddings in the vector store, but the documents contained in the
    document store are returned. This allows image retrieval via the image summaries, while still ensuring
    that the original images associated with the summaries are returned.
  
    Attributes:
        embedding_model (str): Model name used to embed the texts and image summaries.
        store_path (str): Path where the vector and document stores should be saved.
        embeddings (OpenAIEmbeddings): Embedding function used to create vector representations.
        docstore (LocalFileStore): Document store containing texts and images.
        vectorstore (Chroma): Vector store containing embedded texts and image summaries.
        retriever (MultiVectorRetriever): Retriever that encompasses both a vector store and a document store.
        is_new_vectorstore (bool): Flag indicating whether the vector store is new or existing.
        id_key (str): Key used to identify documents.
        doc_ids (list): List of document IDs.
        retrieved_docs (list): List of documents retrieved in the last query.
    """   
    def __init__(self, embedding_model, store_path=None):
        """
        Initialize the SummaryStoreAndRetriever with specified embedding model and storage path.
        
        :param embedding_model: Model name for OpenAI embeddings (e.g., "text-embedding-3-small").
        :param store_path: Path where vector and document stores will be saved.
        """
        
        # Initialize OpenAI embeddings
        print(f"Using OpenAI embedding model: {embedding_model}")
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=os.getenv("OPENAI_API_KEY"),
            chunk_size=64,
            show_progress_bar=True
        )

        self.store_path = store_path
        vectorstore_dir = os.path.join(self.store_path, f"{os.path.basename(self.store_path)}_vectorstore_{embedding_model}")
        docstore_dir = os.path.join(self.store_path, f"{os.path.basename(self.store_path)}_docstore_{embedding_model}")
        
        # Initialize the document store
        self.docstore = LocalFileStore(docstore_dir)
        self.id_key = "doc_id"
        self.doc_ids = []

        # Initialize the vector store with the embedding function
        self.vectorstore = Chroma(
            persist_directory=vectorstore_dir,
            embedding_function=self.embeddings,
            collection_name=f"mm_rag_with_image_summaries_{embedding_model}_embeddings"
        )
        results = self.vectorstore.get(include=["embeddings", "documents", "metadatas"])
        self.is_new_vectorstore = len(results["embeddings"]) > 0

        if self.is_new_vectorstore:
            print(f"Vectorstore at path {vectorstore_dir} already exists")

        else:
            print(f"Creating new vectorstore and docstore at path {self.store_path}")

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key
        )
        self.retrieved_docs = []


    def add_docs(self, doc_summaries: List[str], doc_contents: List[str], doc_filenames: List[str]):
        """
        Add documents to the vector store and document store.
        
        :param doc_summaries: Either text or image summaries to be stored in the vector store.
        :param doc_contents: The original texts or images to be stored in the document store.
        :param doc_filenames: File names associated with the respective documents to be stored as additional metadata.
        """
        if not self.is_new_vectorstore:
            print("Adding documents...")
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={self.id_key: doc_ids[i], "filename": doc_filenames[i]})
                for i, s in enumerate(doc_summaries)
            ]
            self.vectorstore.add_documents(summary_docs)
            self.docstore.mset(list(zip(doc_ids, map(lambda x: str.encode(x), doc_contents))))
        else:
            print("Documents have already been added before, skipping...")


    def retrieve(self, query: str, limit: int) -> List[Document]:
        """
        Retrieve the most relevant documents based on the query.
        """
        self.retrieved_docs = self.retriever.invoke(query, limit=limit)

        return self.retrieved_docs