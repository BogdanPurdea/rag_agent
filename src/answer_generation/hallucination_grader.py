from src.helpers.formatting import format_docs
from src.helpers.config_loader import ConfigLoader
from src.language_models.llm_factory import llm_factory
from langchain_core.messages import HumanMessage, SystemMessage
import json

def get_hallucination_grading_config(key):
    """Fetch a specific key's value from the 'retrieval_grading' section in the configuration."""
    config = ConfigLoader.load_yaml_config("./config/prompt_config.yaml")
    return config.get('hallucination_grading', {}).get(key, "")

def hallucination_grader_instructions():
    return get_hallucination_grading_config('hallucination_grader_instructions')

def hallucination_grader_prompt ():
    return get_hallucination_grading_config('hallucination_grader_prompt')

def hallucination_grader(documents, generation):
    hallucination_grader_prompt_formatted = hallucination_grader_prompt().format(documents=format_docs(documents), generation=generation.content)
    llm_json = llm_factory.get_llm(json_mode=True)
    result = llm_json.invoke([SystemMessage(content=hallucination_grader_instructions())] + [HumanMessage(content=hallucination_grader_prompt_formatted)])
    grade = json.loads(result.content)["binary_score"]
    return grade