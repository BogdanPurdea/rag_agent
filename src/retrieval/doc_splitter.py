from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs_list, chunk_size=1000, chunk_overlap=200):
    """
    Splits documents into smaller chunks.

    Args:
        docs_list (list): List of documents to split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap size between chunks.

    Returns:
        list: A list of split documents.
    """
    print("---SPLITTING DOCUMENTS---")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs_list)
