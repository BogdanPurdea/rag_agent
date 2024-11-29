from langchain_community.document_loaders import WebBaseLoader

def load_documents(urls):
    """
    Loads documents from the provided URLs.

    Args:
        urls (list): List of URLs to load documents from.

    Returns:
        list: A list of loaded documents.
    """
    print("---LOADING DOCUMENTS---")
    docs = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in docs for item in sublist]
