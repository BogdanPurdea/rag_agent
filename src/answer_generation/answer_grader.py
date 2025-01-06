from src.helpers.config_loader import ConfigLoader
from src.language_models.llm_factory import llm_factory
from langchain_core.messages import SystemMessage, HumanMessage
import json
from typing import Any

class AnswerGrader:
    def __init__(self):
        self.config = ConfigLoader.load_yaml_config("./config/prompt_config.yaml")

    def get_answer_grading_config(self, key: str) -> Any:
        """Fetch a specific key's value from the 'retrieval_grading' section in the configuration."""
        return self.config.get('answer_grading', {}).get(key, "")

    def answer_grader_instructions(self) -> str:
        return self.get_answer_grading_config('answer_grader_instructions')

    def answer_grader_prompt(self) -> str:
        return self.get_answer_grading_config('answer_grader_prompt')

    def format_prompt(self, question: str, generation: str) -> str:
        """Format the answer grader prompt with the given question and generation."""
        return self.answer_grader_prompt().format(question=question, generation=generation)

    def grade_answer(self, question: str, generation: Any) -> int:
        answer_grader_prompt_formatted = self.format_prompt(question, generation.content)
        llm_json = llm_factory.get_llm(json_mode=True)
        result = llm_json.invoke([SystemMessage(content=self.answer_grader_instructions())] + [HumanMessage(content=answer_grader_prompt_formatted)])
        grade = json.loads(result.content)["binary_score"]
        return grade