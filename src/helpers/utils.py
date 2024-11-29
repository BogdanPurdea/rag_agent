import json
import yaml

def load_yaml_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Loaded configuration data.
    """
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config
    
def load_json_config(config_path):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Loaded configuration data.
    """
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)