from .doc_loader import load_documents
from .doc_splitter import split_documents
from .vectorstore_setup import setup_vectorstore
import json

def load_config(config_path):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Loaded configuration data.
    """
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config

def create_retriever(config_path):
    """
    Creates and returns a retriever based on the provided URLs.

    Returns:
        retriever: A retriever instance ready to use.
    """
    try:
        # Load URLs from config file
        config = load_config(config_path)
        urls = config.get("urls", [])
        
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
            vectorstore = setup_vectorstore(doc_splits)
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
