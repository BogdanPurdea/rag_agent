from langchain_ollama import ChatOllama

# Configuration: You can store your model info here
LOCAL_LLM = "llama3.2:3b-instruct-fp16"

def get_llm(model=LOCAL_LLM, temperature=0):
    """
    Function to get the LLM model instance.
    
    Parameters:
    - model (str): The model identifier. Default is LOCAL_LLM.
    - temperature (float): Controls randomness in the model's responses.
    
    Returns:
    - ChatOllama instance configured for the LLM.
    """
    llm = ChatOllama(model=model, temperature=temperature)
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
    llm_json_mode = ChatOllama(model=model, temperature=temperature, format="json")
    return llm_json_mode
