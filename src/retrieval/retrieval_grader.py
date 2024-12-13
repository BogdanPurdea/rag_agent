from src.helpers.config_loader import ConfigLoader
from src.language_models.llm_factory import llm_factory
from langchain_core.messages import HumanMessage, SystemMessage
import json

def get_retrieval_grading_config(key):
    """Fetch a specific key's value from the 'retrieval_grading' section in the configuration."""
    config = ConfigLoader.load_yaml_config("./config/prompt_config.yaml")
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
        llm_json = llm_factory.get_llm(json_mode=True)
        result = llm_json.invoke([SystemMessage(content=doc_grader_instructions())] + [HumanMessage(content=doc_grader_prompt_formatted)])
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