from src.helpers.config_loader import ConfigLoader
from src.helpers.date_utils import current_date
from src.language_models.llm_factory import llm_factory
from src.nlp_models.local_llama_model import get_lm_cutoff
from langchain_core.messages import HumanMessage, SystemMessage
import json

# Load routing instructions from config
def router_instructions():
    config = ConfigLoader.load_yaml_config("./config/prompt_config.yaml")
    router_instructions = config.get('routing', {}).get('router_instructions', "")
    return router_instructions

def extract_descriptions(obj):
    descriptions = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "description" and isinstance(value, str):
                descriptions.append(value)
            else:
                descriptions.extend(extract_descriptions(value))
    elif isinstance(obj, list):
        for item in obj:
            descriptions.extend(extract_descriptions(item))
    return descriptions

def get_data_description():
    config = ConfigLoader.load_json_config("./config/data_config.json")
    all_descriptions = extract_descriptions(config)
    return ", ".join(description for description in all_descriptions if description)


def route_to_vectorstore_or_websearch(question):
    router_instructions_formatted = router_instructions().format(description = get_data_description(), cutoff_date= get_lm_cutoff("llama3.2:1b"), current_date = current_date())
    llm_json = llm_factory.get_llm(json_mode=True)
    route_question = llm_json.invoke([SystemMessage(content=router_instructions_formatted)] + [HumanMessage(content=question)])
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    
def route_to_generate_or_websearch(web_search):
    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"