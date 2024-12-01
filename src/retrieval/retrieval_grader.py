from src.helpers.utils import load_yaml_config

def get_retrieval_grading_config(key):
    """Fetch a specific key's value from the 'retrieval_grading' section in the configuration."""
    config = load_yaml_config("./config/prompt_config.yaml")
    return config.get('retrieval_grading', {}).get(key, "")

def doc_grader_instructions():
    return get_retrieval_grading_config('doc_grader_instructions')

def doc_grader_prompt():
    return get_retrieval_grading_config('doc_grader_prompt')
