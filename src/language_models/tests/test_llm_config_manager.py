import os
import json
import unittest
from unittest.mock import patch, mock_open
from langchain_ollama import ChatOllama

from src.language_models.llm_config_manager import LLMConfigManager, LLMConfigurationError

class TestLLMConfigManager(unittest.TestCase):
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.mock_config_data = {
            "local_based": {
                "llama3.2:1b": {
                    "model_name": "llama3.2-instruct-fp16",
                    "cutoff_date": "December 2023"
                },
                "mistral:7b": {
                    "model_name": "mistral-7b-instruct",
                    "cutoff_date": "2024-06-15"
                }
            }
        }
        
        # Create a mock configuration file path
        self.mock_config_path = "./config/test_lm_config.json"
        with open(self.mock_config_path, 'w') as f:
            json.dump(self.mock_config_data, f)

    def tearDown(self):
        """
        Clean up after each test method.
        """
        # Remove the temporary config file
        if os.path.exists(self.mock_config_path):
            os.remove(self.mock_config_path)

    def test_successful_config_loading(self):
        """
        Test that configuration is loaded successfully.
        """
        config_manager = LLMConfigManager(self.mock_config_path)
        
        # Verify configuration is loaded correctly
        self.assertEqual(
            config_manager._config, 
            self.mock_config_data
        )

    def test_get_model_config_successful(self):
        """
        Test retrieving model configuration for existing models.
        """
        config_manager = LLMConfigManager(self.mock_config_path)
        
        # Test llama model configuration
        llama_config = config_manager.get_model_config("llama3.2:1b")
        self.assertEqual(llama_config, {
            "model_name": "llama3.2-instruct-fp16",
            "cutoff_date": "December 2023"
        })
        
        # Test mistral model configuration
        mistral_config = config_manager.get_model_config("mistral:7b")
        self.assertEqual(mistral_config, {
            "model_name": "mistral-7b-instruct",
            "cutoff_date": "2024-06-15"
        })

    def test_get_model_config_non_existent(self):
        """
        Test error handling for non-existent model configurations.
        """
        config_manager = LLMConfigManager(self.mock_config_path)
        
        with self.assertRaises(LLMConfigurationError) as context:
            config_manager.get_model_config("unknown_model")
        
        self.assertIn(
            "No configuration found for model: unknown_model", 
            str(context.exception)
        )

    def test_create_llm_successful(self):
        """
        Test successful LLM creation with various parameters.
        """
        config_manager = LLMConfigManager(self.mock_config_path)
        
        # Test default model creation
        llm = config_manager.create_llm()
        self.assertIsInstance(llm, ChatOllama)
        self.assertEqual(llm.model, "llama3.2-instruct-fp16")
        self.assertEqual(llm.temperature, 0.0)
        
        # Test custom model with JSON mode
        llm_custom = config_manager.create_llm(
            model="mistral:7b", 
            temperature=0.5, 
            json_mode=True
        )
        self.assertIsInstance(llm_custom, ChatOllama)
        self.assertEqual(llm_custom.model, "mistral-7b-instruct")
        self.assertEqual(llm_custom.temperature, 0.5)
        self.assertEqual(llm_custom.format, "json")

    def test_create_llm_invalid_temperature(self):
        """
        Test error handling for invalid temperature values.
        """
        config_manager = LLMConfigManager(self.mock_config_path)
        
        # Test negative temperature
        with self.assertRaises(ValueError) as context_low:
            config_manager.create_llm(temperature=-0.1)
        self.assertIn(
            "Temperature must be between 0 and 1", 
            str(context_low.exception)
        )
        
        # Test temperature above 1
        with self.assertRaises(ValueError) as context_high:
            config_manager.create_llm(temperature=1.1)
        self.assertIn(
            "Temperature must be between 0 and 1", 
            str(context_high.exception)
        )

    @patch('src.helpers.utils.load_json_config')
    def test_configuration_loading_error(self, mock_load_config):
        """
        Test error handling during configuration loading.
        """
        # Simulate empty configuration
        mock_load_config.return_value = {}
        
        with self.assertRaises(LLMConfigurationError) as context_empty:
            LLMConfigManager("dummy_path.json")
        self.assertIn(
            "Failed to load LLM configuration:", 
            str(context_empty.exception)
        )
        
        # Simulate loading exception
        mock_load_config.side_effect = Exception("File not found")
        
        with self.assertRaises(LLMConfigurationError) as context_error:
            LLMConfigManager("non_existent_path.json")
        self.assertIn(
            "Failed to load LLM configuration", 
            str(context_error.exception)
        )

if __name__ == '__main__':
    unittest.main()