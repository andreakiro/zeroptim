from pydantic import ValidationError
from zeroptim.configs._types import Config
import yaml

def parse(filepath: str):
    assert filepath.endswith(('.yaml', '.yml')), "File extension must be .yaml"
    with open(filepath, 'r') as file:
        config_dict = yaml.safe_load(file)
    try:
        validated_config = Config(**config_dict)
        return validated_config
    except ValidationError as e:
        print(f"Error in configuration: {e}")
        raise

def dump(config: Config, filepath: str) -> None:
    assert filepath.endswith(('.yaml', '.yml')), "File extension must be .yaml"
    assert config is not None and isinstance(config, Config), "A Config is required"
    yaml_str = yaml.dump(config.model_dump(), sort_keys=False)
    with open(filepath, 'w') as file:
        file.write(yaml_str)
