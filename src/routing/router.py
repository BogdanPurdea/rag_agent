from src.helpers.utils import load_yaml_config

# Load routing instructions from config
def router_instructions():
    config = load_yaml_config("./config/prompt_config.yaml")
    router_instructions = config.get('routing', {}).get('router_instructions', "")
    return router_instructions