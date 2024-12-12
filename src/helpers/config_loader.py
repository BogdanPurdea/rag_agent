import json
import yaml
from typing import Callable

class ConfigLoader:
    """
    A class for loading configuration files in various formats.
    """
    
    @staticmethod
    def load_config(config_path: str, parser: Callable) -> dict:
        """
        Load configuration from a file using a specified parser.

        Args:
            config_path (str): Path to the configuration file.
            parser (Callable): A function to parse the file content.

        Returns:
            dict: Loaded configuration data.
        """
        with open(config_path, "r") as config_file:
            return parser(config_file)
    
    @classmethod
    def load_json_config(cls, config_path: str) -> dict:
        """
        Load configuration from a JSON file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            dict: Loaded configuration data.
        """
        return cls.load_config(config_path, json.load)
    
    @classmethod
    def load_yaml_config(cls, config_path: str) -> dict:
        """
        Load configuration from a YAML file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            dict: Loaded configuration data.
        """
        return cls.load_config(config_path, yaml.safe_load)
