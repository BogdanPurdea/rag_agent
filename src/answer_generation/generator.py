from src.helpers.formatting import format_docs
from src.helpers.config_loader import ConfigLoader
from langchain_core.messages import HumanMessage
from src.language_models.llm_factory import llm_factory
from src.answer_generation.hallucination_grader import hallucination_grader
from src.answer_generation.answer_grader import answer_grader

def rag_prompt():
    config = ConfigLoader.load_yaml_config("./config/prompt_config.yaml")
    rag_prompt = config.get('generation', {}).get('rag_prompt', "")
    return rag_prompt

def generate_answer(documents, question):
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt().format(context=docs_txt, question=question)
    llm = llm_factory.get_llm()
    response = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return response

def grade_generation(documents, question, generation, loop_step, max_retries):
    hallucination_grade = hallucination_grader(documents, generation)
    if hallucination_grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above
        answer_grade = answer_grader(question, generation)
        if answer_grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif loop_step <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif loop_step <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"