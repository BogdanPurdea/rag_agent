from src.helpers.config_loader import ConfigLoader
from src.language_models.llm_factory import llm_factory
from langchain_core.messages import SystemMessage, HumanMessage
import json

def get_answer_grading_config(key):
    """Fetch a specific key's value from the 'retrieval_grading' section in the configuration."""
    config = ConfigLoader.load_yaml_config("./config/prompt_config.yaml")
    return config.get('answer_grading', {}).get(key, "")

def answer_grader_instructions():
    return get_answer_grading_config('answer_grader_instructions')

def answer_grader_prompt():
    return get_answer_grading_config('answer_grader_prompt')

def answer_grader(question, generation):
    answer_grader_prompt_formatted = answer_grader_prompt().format(question=question, generation=generation.content)
    llm_json = llm_factory.get_llm(json_mode=True)
    result = llm_json.invoke([SystemMessage(content=answer_grader_instructions())] + [HumanMessage(content=answer_grader_prompt_formatted)])
    grade = json.loads(result.content)["binary_score"]
    return grade