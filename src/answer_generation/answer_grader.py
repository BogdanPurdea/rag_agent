from src.helpers.utils import load_yaml_config
from src.nlp_models.local_llama_model import get_llm_json_mode
from langchain_core.messages import SystemMessage, HumanMessage
import json

def get_answer_grading_config(key):
    """Fetch a specific key's value from the 'retrieval_grading' section in the configuration."""
    config = load_yaml_config("./config/prompt_config.yaml")
    return config.get('answer_grading', {}).get(key, "")

def answer_grader_instructions():
    return get_answer_grading_config('answer_grader_instructions')

def answer_grader_prompt():
    return get_answer_grading_config('answer_grader_prompt')

def answer_grader(question, generation):
    answer_grader_prompt_formatted = answer_grader_prompt().format(question=question, generation=generation.content)
    result = get_llm_json_mode().invoke([SystemMessage(content=answer_grader_instructions())] + [HumanMessage(content=answer_grader_prompt_formatted)])
    grade = json.loads(result.content)["binary_score"]
    return grade