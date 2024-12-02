from src.helpers.utils import load_yaml_config
from src.nlp_models.local_llama_model import get_llm_json_mode
from langchain_core.messages import HumanMessage, SystemMessage
import json

def get_retrieval_grading_config(key):
    """Fetch a specific key's value from the 'retrieval_grading' section in the configuration."""
    config = load_yaml_config("./config/prompt_config.yaml")
    return config.get('retrieval_grading', {}).get(key, "")

def doc_grader_instructions():
    return get_retrieval_grading_config('doc_grader_instructions')

def doc_grader_prompt():
    return get_retrieval_grading_config('doc_grader_prompt')

def doc_grade(question, documents):
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt().format(document=d.page_content, question=question)
        result = get_llm_json_mode().invoke([SystemMessage(content=doc_grader_instructions())] + [HumanMessage(content=doc_grader_prompt_formatted)])
        grade = json.loads(result.content)["binary_score"]
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return filtered_docs, web_search