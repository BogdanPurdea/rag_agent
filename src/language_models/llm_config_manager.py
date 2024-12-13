from typing import Dict
from functools import lru_cache

from langchain_ollama import ChatOllama
from src.helpers.config_loader import ConfigLoader

class LLMConfigurationError(Exception):
    """Custom exception for LLM configuration errors."""
    pass

class LLMConfigManager:
    """Manages configuration and instantiation of Language Models."""
    
    def __init__(self, config_path: str = "./config/lm_config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str): Path to the LLM configuration JSON file.
        """
        self._config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict:
        """
        Load and validate the configuration.
        
        Returns:
            Dict: Validated configuration dictionary.
        
        Raises:
            LLMConfigurationError: If configuration loading or validation fails.
        """
        try:
            config = ConfigLoader.load_json_config(self._config_path)
            if not config or 'local_based' not in config:
                raise LLMConfigurationError("Invalid or empty LLM configuration")
            return config
        except Exception as e:
            raise LLMConfigurationError(f"Failed to load LLM configuration: {e}")
    
    @lru_cache(maxsize=32)
    def get_model_config(self, model: str = "llama3.2:1b") -> Dict:
        """
        Retrieve configuration for a specific model.
        
        Args:
            model (str): Model identifier.
        
        Returns:
            Dict: Model-specific configuration.
        
        Raises:
            LLMConfigurationError: If model configuration is not found.
        """
        local_configs = self._config.get('local_based', {})
        model_config = local_configs.get(model)
        
        if not model_config:
            raise LLMConfigurationError(f"No configuration found for model: {model}")
        
        return model_config
    
    def create_llm(
        self, 
        model: str = "llama3.2:1b", 
        temperature: float = 0.0, 
        json_mode: bool = False
    ) -> ChatOllama:
        """
        Create a configured ChatOllama instance.
        
        Args:
            model (str): Model identifier.
            temperature (float): Response randomness control.
            json_mode (bool): Whether to enable JSON output mode.
        
        Returns:
            ChatOllama: Configured language model instance.
        """
        # Validate temperature
        if not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        
        # Get model name from configuration
        model_name = self.get_model_config(model).get('model_name', '')
        
        # Create and return ChatOllama instance
        return ChatOllama(
            model=model_name, 
            temperature=temperature,
            format="json" if json_mode else ""
        )
