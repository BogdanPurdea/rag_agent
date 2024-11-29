from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings

def setup_vectorstore(doc_splits, model_name="nomic-embed-text-v1.5"):
    """
    Sets up a vector store and retriever from split documents.

    Args:
        doc_splits (list): List of split documents.
        model_name (str): The name of the embedding model.

    Returns:
        retriever: The retriever instance from the vector store.
    """
    print("---INITIALIZING VECTORSTORE---")

    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=NomicEmbeddings(model=model_name, inference_mode="local"),
    )

    return vectorstore
