import unittest
from unittest.mock import MagicMock
import json

from src.routing.router import router_instructions
from src.nlp_models.local_llama_model import get_llm_json_mode
from langchain_core.messages import HumanMessage, SystemMessage


class TestRouter(unittest.TestCase):

    router_instructions = router_instructions()

    def setUp(self):
        # Mock the LLM's invoke method
        self.mock_invoke = MagicMock()
        get_llm_json_mode.invoke = self.mock_invoke

    def test_web_search_routing(self):
        # Simulating a question that should be routed to web search
        self.mock_invoke.return_value.content = json.dumps({"datasource": "websearch"})
        
        response = get_llm_json_mode.invoke(
            [SystemMessage(content=self.router_instructions)]
            + [HumanMessage(content="Who is favored to win the NFC Championship game in the 2024 season?")]
        )

        # Test that the response is routed to the correct datasource
        result = json.loads(response.content)
        self.assertEqual(result["datasource"], "websearch")

    def test_vector_store_routing(self):
        # Simulating a question that should be routed to vectorstore
        self.mock_invoke.return_value.content = json.dumps({"datasource": "vectorstore"})
        
        response = get_llm_json_mode.invoke(
            [SystemMessage(content=self.router_instructions)]
            + [HumanMessage(content="What are the types of agent memory?")]
        )

        # Test that the response is routed to the correct datasource
        result = json.loads(response.content)
        self.assertEqual(result["datasource"], "vectorstore")

    def test_mixed_routing(self):
        # Simulating a mixed test scenario with both web search and vector store
        self.mock_invoke.return_value.content = json.dumps({"datasource": "websearch"})
        
        # Test Web search routing
        response_web_search = get_llm_json_mode.invoke(
            [SystemMessage(content=self.router_instructions)]
            + [HumanMessage(content="Who is favored to win the NFC Championship game in the 2024 season?")]
        )
        result_web_search = json.loads(response_web_search.content)
        self.assertEqual(result_web_search["datasource"], "websearch")

        self.mock_invoke.return_value.content = json.dumps({"datasource": "vectorstore"})

        # Test Vector store routing
        response_vector_store = get_llm_json_mode.invoke(
            [SystemMessage(content=self.router_instructions)]
            + [HumanMessage(content="What are the types of agent memory?")]
        )
        result_vector_store = json.loads(response_vector_store.content)
        self.assertEqual(result_vector_store["datasource"], "vectorstore")

    def tearDown(self):
        # Clean up any necessary resources after each test
        pass


if __name__ == "__main__":
    unittest.main()
