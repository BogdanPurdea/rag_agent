from src.helpers.utils import load_yaml_config

def get_hallucination_grading_config(key):
    """Fetch a specific key's value from the 'retrieval_grading' section in the configuration."""
    config = load_yaml_config("./config/prompt_config.yaml")
    return config.get('hallucination_grading', {}).get(key, "")

def hallucination_grader_instructions():
    # return get_hallucination_grading_config('hallucination_grader_instructions')
    return """You are a teacher grading a quiz. 

        You will be given FACTS and a STUDENT ANSWER. 

        Here is the grade criteria to follow:

        (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

        (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

        Score:

        A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

        A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

        Avoid simply stating the correct answer at the outset."""


def hallucination_grader_prompt ():
    # return get_hallucination_grading_config('hallucination_grader_prompt ')
    return  """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 
        Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""
