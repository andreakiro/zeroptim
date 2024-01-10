from zeroptim.configs.serialization import parse, dump
from zeroptim.configs._types import Config
import os

def resolve(s: str) -> str:
    s = s if s.startswith("configs/") else f"configs/{s}"
    s = s[:-5] if s.endswith('.yaml') else s
    s = s[:-4] if s.endswith('.yml') else s
    if os.path.exists(f"{s}.yaml"): return f"{s}.yaml"
    if os.path.exists(f"{s}.yml"): return f"{s}.yml"
    raise FileNotFoundError(f"Could not find config file {s}")

def load(filepath: str) -> Config:
    # alias for parse
    return parse(resolve(filepath))

def save(config: Config, filepath: str) -> None:
    # alias for dump
    return dump(config, filepath)
