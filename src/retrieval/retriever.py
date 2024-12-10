from .doc_loader import load_documents
from .doc_splitter import split_documents
from .vectorstore_setup import setup_vectorstore
from src.helpers.utils import load_json_config
import json

def extract_urls(obj):
    urls = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "urls" and isinstance(value, list):
                urls.extend(value)
            else:
                urls.extend(extract_urls(value))
    elif isinstance(obj, list):
        for item in obj:
            urls.extend(extract_urls(item))
    return urls

def create_retriever(config_path):
    """
    Creates and returns a retriever based on the provided URLs.

    Returns:
        retriever: A retriever instance ready to use.
    """
    try:
        # Load URLs from config file
        config = load_json_config(config_path)
        persist_path = config.get("vectorstore", {}).get("persist_path", "./data/vectorstore.json")
        model_name = config.get("vectorstore", {}).get("embedding_model", "nomic-embed-text")
        urls = extract_urls(config)
        
        # Load documents
        try:
            docs_list = load_documents(urls)
        except Exception as e:
            raise RuntimeError(f"Error loading documents: {e}")

        # Split documents
        try:
            doc_splits = split_documents(docs_list)
        except Exception as e:
            raise RuntimeError(f"Error splitting documents: {e}")

        # Setup vector store
        try:
            vectorstore = setup_vectorstore(doc_splits, model_name, persist_path)
        except Exception as e:
            raise RuntimeError(f"Error setting up vector store: {e}")

        # Create retriever
        try:
            retriever = vectorstore.as_retriever(k=3)
        except Exception as e:
            raise RuntimeError(f"Error creating retriever: {e}")

        return retriever

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing config file '{config_path}'. Ensure it is valid JSON.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

def retrieve_documents(question, config_path="./config/data_config.json"):
    retriever = create_retriever(config_path)
    return retriever.invoke(question)