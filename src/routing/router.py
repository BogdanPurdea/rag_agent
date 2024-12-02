from src.helpers.utils import load_yaml_config
from src.helpers.utils import current_date
from src.nlp_models.local_llama_model import get_llm_json_mode
from langchain_core.messages import HumanMessage, SystemMessage
import json

# Load routing instructions from config
def router_instructions():
    config = load_yaml_config("./config/prompt_config.yaml")
    router_instructions = config.get('routing', {}).get('router_instructions', "")
    return router_instructions + current_date() + "\""

def route_to_vectorstore_or_websearch(question):
    route_question = get_llm_json_mode().invoke([SystemMessage(content=router_instructions())] + [HumanMessage(content=question)])
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