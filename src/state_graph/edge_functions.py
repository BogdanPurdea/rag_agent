# edge_functions.py
from typing import Dict
import json

def route_question(state: Dict):
    """Route question to web search or RAG"""
    print("---ROUTE QUESTION---")
    route_question = llm_json_mode.invoke([SystemMessage(content=router_instructions)] + [HumanMessage(content=state["question"])])
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        return "websearch"
    elif source == "vectorstore":
        return "vectorstore"

def decide_to_generate(state: Dict):
    """Determine whether to generate an answer, or add web search"""
    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    if web_search == "Yes":
        return "websearch"
    else:
        return "generate"

def grade_generation_v_documents_and_question(state: Dict):
    """Check if the generation is grounded in the document and answers the question"""
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(documents=format_docs(documents), generation=generation.content)
    result = llm_json_mode.invoke([SystemMessage(content=hallucination_grader_instructions)] + [HumanMessage(content=hallucination_grader_prompt_formatted)])
    grade = json.loads(result.content)["binary_score"]
    if grade == "yes":
        answer_grader_prompt_formatted = answer_grader_prompt.format(question=question, generation=generation.content)
        result = llm_json_mode.invoke([SystemMessage(content=answer_grader_instructions)] + [HumanMessage(content=answer_grader_prompt_formatted)])
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            return "useful"
        elif state["loop_step"] <= max_retries:
            return "not useful"
        else:
            return "max retries"
    elif state["loop_step"] <= max_retries:
        return "not supported"
    else:
        return "max retries"
