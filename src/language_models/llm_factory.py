from typing import Dict, Optional

from langchain_ollama import ChatOllama
from src.language_models.llm_config_manager import LLMConfigManager

class LLMFactory:
    """
    A factory class for creating and managing LLM instances.
    
    This class ensures that only one instance of each LLM configuration 
    is created and reused across the application.
    """
    
    def __init__(self, config_manager: Optional[LLMConfigManager] = None):
        """
        Initialize the LLM Factory.
        
        Args:
            config_manager (Optional[LLMConfigManager]): 
                Configuration manager for LLM settings. 
                If not provided, a default instance will be created.
        """
        self._config_manager = config_manager or LLMConfigManager()
        
        # Dictionary to store created LLM instances
        self._llm_instances: Dict[str, Dict[tuple, ChatOllama]] = {}
    
    def get_llm(
        self, 
        model: str = "llama3.2:1b", 
        temperature: float = 0.0, 
        json_mode: bool = False
    ) -> ChatOllama:
        """
        Get or create an LLM instance with specific configuration.
        
        Args:
            model (str): Model identifier.
            temperature (float): Response randomness control.
            json_mode (bool): Whether to enable JSON output mode.
        
        Returns:
            ChatOllama: Configured language model instance.
        """
        # Create a unique key for the LLM configuration
        config_key = (model, temperature, json_mode)
        
        # Check if the LLM instance already exists
        if model not in self._llm_instances:
            self._llm_instances[model] = {}
        
        # Check if the specific configuration exists
        if config_key not in self._llm_instances[model]:
            # Create a new LLM instance
            self._llm_instances[model][config_key] = self._config_manager.create_llm(
                model=model,
                temperature=temperature,
                json_mode=json_mode
            )
        return self._llm_instances[model][config_key]
    
    def clear_cache(self):
        """
        Clear all cached LLM instances.
        Useful for resetting or refreshing LLM configurations.
        """
        self._llm_instances.clear()

# Create a singleton instance for global use
llm_factory = LLMFactory()