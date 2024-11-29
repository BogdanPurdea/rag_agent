# graph_setup.py
from langgraph.graph import StateGraph
from .state import GraphState
from .node_functions import retrieve, generate, grade_documents, web_search
from .edge_functions import route_question, decide_to_generate, grade_generation_v_documents_and_question
from langgraph.graph import END

def setup_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    # Set conditional entry points and edges
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
            "max retries": END,
        },
    )

    # Compile the graph
    graph = workflow.compile()
    return graph
