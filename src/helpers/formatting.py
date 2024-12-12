def format_docs(docs) -> str:
    """
    Format document contents into a single string with double line breaks.

    Args:
        docs (list): List of documents, each with a 'page_content' attribute.

    Returns:
        str: Concatenated document contents separated by double line breaks.
    """
    return "\n\n".join(doc.page_content for doc in docs)
