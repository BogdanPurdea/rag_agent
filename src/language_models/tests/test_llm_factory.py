import unittest
from unittest.mock import Mock
from langchain_ollama import ChatOllama

from src.language_models.llm_factory import LLMFactory
from src.language_models.llm_config_manager import LLMConfigManager

class TestLLMFactory(unittest.TestCase):
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        # Create a mock config manager
        self.mock_config_manager = Mock(spec=LLMConfigManager)
        
        # Create a mock ChatOllama instance
        self.mock_llm = Mock(spec=ChatOllama)
        
        # Configure the mock config manager to return the mock LLM
        self.mock_config_manager.create_llm.return_value = self.mock_llm
        
        # Create the factory with the mock config manager
        self.factory = LLMFactory(self.mock_config_manager)

    def test_get_llm_creates_instance(self):
        """
        Test that get_llm creates a new LLM instance.
        """
        # Define the expected configuration
        model = "test_model"
        temperature = 0.0
        json_mode = False
        config_key = (model, temperature, json_mode)

        # Call get_llm
        llm = self.factory.get_llm(model=model)
        
        # Verify the config manager's create_llm was called
        self.mock_config_manager.create_llm.assert_called_once_with(
            model=model,
            temperature=temperature,
            json_mode=json_mode
        )
        
        # Verify the returned instance is the mocked LLM
        self.assertEqual(llm, self.mock_llm)
        
        # Verify the internal structure of _llm_instances
        self.assertIn(model, self.factory._llm_instances)
        self.assertIn(config_key, self.factory._llm_instances[model])

    def test_get_llm_caches_instance(self):
        """
        Test that subsequent calls return the same LLM instance.
        """
        # Define the expected configuration
        model = "test_model"

        # First call
        first_llm = self.factory.get_llm(model=model)
        
        # Reset the mock to check how many times it was called
        self.mock_config_manager.create_llm.reset_mock()
        
        # Second call
        second_llm = self.factory.get_llm(model=model)
        
        # Verify create_llm was not called again
        self.mock_config_manager.create_llm.assert_not_called()
        
        # Verify both instances are the same
        self.assertEqual(first_llm, second_llm)

    def test_get_llm_different_configurations(self):
        """
        Test that different configurations create different instances.
        """
        # Create LLMs with different configurations
        llm_default = self.factory.get_llm(model="test_model")
        llm_json = self.factory.get_llm(model="test_model", json_mode=True)
        llm_temp = self.factory.get_llm(model="test_model", temperature=0.5)
        
        # Verify create_llm was called with correct parameters
        calls = self.mock_config_manager.create_llm.call_args_list
        self.assertEqual(len(calls), 3)
        
        # Check specific call arguments
        self.assertEqual(
            calls[0][1], 
            {'model': 'test_model', 'temperature': 0.0, 'json_mode': False}
        )
        self.assertEqual(
            calls[1][1], 
            {'model': 'test_model', 'temperature': 0.0, 'json_mode': True}
        )
        self.assertEqual(
            calls[2][1], 
            {'model': 'test_model', 'temperature': 0.5, 'json_mode': False}
        )

    def test_clear_cache(self):
        """
        Test that clear_cache removes all cached instances.
        """
        # Create some LLM instances
        self.factory.get_llm(model="model1")
        self.factory.get_llm(model="model2", json_mode=True)
        
        # Reset the mock to track new calls
        self.mock_config_manager.create_llm.reset_mock()
        
        # Clear the cache
        self.factory.clear_cache()
        
        # Create instances again
        self.factory.get_llm(model="model1")
        self.factory.get_llm(model="model2", json_mode=True)
        
        # Verify create_llm was called for both instances
        self.assertEqual(self.mock_config_manager.create_llm.call_count, 2)

    def test_default_config_manager(self):
        """
        Test that a default config manager is created if not provided.
        """
        # Create factory without a config manager
        factory = LLMFactory()
        
        # Verify it creates an LLM without errors
        llm = factory.get_llm()
        self.assertIsNotNone(llm)

    def test_multiple_models(self):
        """
        Test creating LLMs for multiple different models.
        """
        # Create LLMs for different models
        llm1 = self.factory.get_llm(model="model1")
        llm2 = self.factory.get_llm(model="model2")
        
        # Verify different models create different instances
        self.assertEqual(
            self.mock_config_manager.create_llm.call_count, 
            2
        )
        
        # Verify the instances are distinct
        # self.assertNotEqual(llm1, llm2)

if __name__ == '__main__':
    unittest.main()