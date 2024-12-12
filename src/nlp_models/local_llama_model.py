from langchain_ollama import ChatOllama
from src.helpers.config_loader import ConfigLoader
# Configuration: You can store your model info here
LOCAL_LLM = "llama3.2:1b"
def get_local_lm_config(model=LOCAL_LLM):
    config = ConfigLoader.load_json_config("./config/lm_config.json")
    return config.get("local_based", {}).get(model, {})

def get_lm_name(model):
    return get_local_lm_config(model).get('model_name', "")

def get_lm_cutoff(model):
    return get_local_lm_config(model).get('cutoff_date', "")

def get_llm(model=LOCAL_LLM, temperature=0):
    """
    Function to get the LLM model instance.
    
    Parameters:
    - model (str): The model identifier. Default is LOCAL_LLM.
    - temperature (float): Controls randomness in the model's responses.
    
    Returns:
    - ChatOllama instance configured for the LLM.
    """
    llm = ChatOllama(model=get_lm_name(model), temperature=temperature)
    return llm

def get_llm_json_mode(model=LOCAL_LLM, temperature=0):
    """
    Function to get the LLM model instance configured for JSON format.
    
    Parameters:
    - model (str): The model identifier. Default is LOCAL_LLM.
    - temperature (float): Controls randomness in the model's responses.
    
    Returns:
    - ChatOllama instance configured for the LLM in JSON format.
    """
    llm_json_mode = ChatOllama(model=get_lm_name(model), temperature=temperature, format="json")
    return llm_json_mode
