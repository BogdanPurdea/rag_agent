# node_functions.py
from typing import Dict
from langchain.schema import Document
from src.retrieval.retriever import create_retriever
def retrieve(state: Dict):
    """Retrieve documents from vectorstore"""
    print("---RETRIEVE---")
    question = state["question"]
    retriever = create_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents}

def generate(state: Dict):
    """Generate answer using RAG on retrieved documents"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}

def grade_documents(state: Dict):
    """Determine whether the retrieved documents are relevant to the question"""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=d.page_content, question=question)
        result = llm_json_mode.invoke([SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])
        grade = json.loads(result.content)["binary_score"]
        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
    return {"documents": filtered_docs, "web_search": web_search}

def web_search(state: Dict):
    """Perform a web search based on the question"""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}
