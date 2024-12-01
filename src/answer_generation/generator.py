from src.helpers.utils import load_yaml_config

def rag_prompt():
    config = load_yaml_config("./config/prompt_config.yaml")
    rag_prompt = config.get('generation', {}).get('rag_prompt', "")
    return rag_prompt