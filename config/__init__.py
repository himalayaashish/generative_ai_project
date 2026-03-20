import yaml
import os

def load_config(config_name: str):
    config_path = os.path.join(os.path.dirname(__file__), f"{config_name}.yaml")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Pre-load for easy access
MODEL_CONFIG = load_config("model_config")
PROMPT_TEMPLATES = load_config("prompt_templates")