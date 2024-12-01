from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings
import os

def setup_vectorstore(doc_splits, model_name="nomic-embed-text"):
    """
    Sets up a vector store and retriever from split documents.

    Args:
        doc_splits (list): List of split documents.
        model_name (str): The name of the embedding model.

    Returns:
        vector store: The vector store instance using the document splits and embedding model provided.
    """
    print("---INITIALIZING VECTORSTORE---")

    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=OllamaEmbeddings(model=model_name),
    )
    return vectorstore


# TO DO - VERIFY AND FIX LOCAL LOADING OF VECTORSTORE
def load_vectorstore(doc_splits, model_name = "nomic-embed-text-v1.5"):
    
    vectorstore_path = "path/to/vectorstore.pkl"  # Path to save/load the vectorstore

    # Check if the vectorstore already exists locally
    if os.path.exists(vectorstore_path):
        vector_store = SKLearnVectorStore(
            embedding=NomicEmbeddings(model=model_name, inference_mode="local"),
            persist_path=vectorstore_path,
        )
    else:
        print("Initializing new vectorstore...")
        vectorstore = setup_vectorstore(doc_splits, model_name)
        # Save the vectorstore locally for future use
        vectorstore.persist()
        print(f"Vector store saved to {vectorstore_path}")
