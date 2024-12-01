from src.helpers.utils import load_yaml_config

def get_answer_grading_config(key):
    """Fetch a specific key's value from the 'retrieval_grading' section in the configuration."""
    config = load_yaml_config("./config/prompt_config.yaml")
    return config.get('answer_grading', {}).get(key, "")

def answer_grader_instructions():
    return get_answer_grading_config('answer_grader_instructions')

def answer_grader_prompt ():
    return get_answer_grading_config('answer_grader_prompt ')
