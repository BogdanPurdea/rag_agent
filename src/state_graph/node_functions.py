# node_functions.py
from typing import Dict
from src.retrieval.retriever import retrieve_documents
from src.retrieval.retrieval_grader import doc_grade
from src.answer_generation.generator import generate_answer
from src.web_search.web_search_tool import web_search

def retrieve(state: Dict):
    """Retrieve documents from vectorstore"""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retrieve_documents(question, "./config/data_config.json")
    return {"documents": documents}

def generate(state: Dict):
    """Generate answer using RAG on retrieved documents"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    generation = generate_answer(documents, question)
    return {"generation": generation, "loop_step": loop_step + 1}

def grade_documents(state: Dict):
    """Determine whether the retrieved documents are relevant to the question"""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs, web_search = doc_grade(question, documents)
    return {"documents": filtered_docs, "web_search": web_search}

def web_search(state: Dict):
    """Perform a web search based on the question"""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    web_results = web_search(question)
    documents.append(web_results)
    return {"documents": documents}
