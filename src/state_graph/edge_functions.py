# edge_functions.py
from src.answer_generation.generator import grade_generation
from src.routing.router import route_to_vectorstore_or_websearch, route_to_generate_or_websearch
from typing import Dict


def route_question(state: Dict):
    """Route question to web search or RAG"""
    print("---ROUTE QUESTION---")
    question = state["question"]
    return route_to_vectorstore_or_websearch(question)

def decide_to_generate(state: Dict):
    """Determine whether to generate an answer, or add web search"""
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    return route_to_generate_or_websearch(web_search)

def grade_generation_v_documents_and_question(state: Dict):
    """Check if the generation is grounded in the document and answers the question"""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    loop_step = state["loop_step"]
    max_retries = state.get("max_retries", 3)
    return grade_generation(documents, question, generation, loop_step, max_retries)
