from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings
import os

def create_vectorstore(doc_splits, model_name="nomic-embed-text", persist_path="./data/vectorstore.pkl"):
    """
    Creates a new vector store from split documents and saves it locally.

    Args:
        doc_splits (list): List of split documents.
        model_name (str): The name of the embedding model.
        persist_path (str): Path to save the vectorstore.

    Returns:
        SKLearnVectorStore: The newly created and saved vector store.
    """
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)

    print("---CREATING NEW VECTORSTORE---")
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=OllamaEmbeddings(model=model_name),
        persist_path = persist_path
    )

    # Persist the vectorstore
    vectorstore.persist()
    print(f"Vector store saved to {persist_path}")
    
    return vectorstore

def load_vectorstore(model_name="nomic-embed-text", persist_path="./data/vectorstore.pkl"):
    """
    Loads an existing vectorstore from a local file.

    Args:
        model_name (str): The name of the embedding model.
        persist_path (str): Path to the saved vectorstore.

    Returns:
        SKLearnVectorStore: The loaded vector store.
    """
    if not os.path.exists(persist_path):
        raise FileNotFoundError(f"No vectorstore found at {persist_path}")

    print("---LOADING EXISTING VECTORSTORE---")
    vectorstore = SKLearnVectorStore(
        embedding=OllamaEmbeddings(model=model_name),
        persist_path=persist_path,
    )
    
    return vectorstore

def setup_vectorstore(doc_splits, model_name="nomic-embed-text", persist_path="./data/vectorstore.pkl"):
    """
    Conditionally creates or loads a vectorstore based on existing data.

    Args:
        doc_splits (list): List of split documents.
        model_name (str): The name of the embedding model.
        persist_path (str): Path to save/load the vectorstore.

    Returns:
        SKLearnVectorStore: The vector store (either loaded or newly created).
    """
    try:
        # Try to load existing vectorstore
        return load_vectorstore(model_name, persist_path)
    except FileNotFoundError:
        # If no existing vectorstore, create a new one
        return create_vectorstore(doc_splits, model_name, persist_path)
    
def clean_vectorstore(persist_path):
    os.remove(persist_path)