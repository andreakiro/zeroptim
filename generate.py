import yaml
from itertools import product
from pathlib import Path
import copy


from zeroptim.configs import Config
from zeroptim.configs.serialization import dump

SWEEP_KEY = "SWEEP."


def load_sweep(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def find_sweep_groups(d, path=None):
    if path is None:
        path = []

    for k, v in d.items():
        new_path = path + [k]
        if isinstance(v, dict):
            yield from find_sweep_groups(v, new_path)
        elif SWEEP_KEY in k:
            yield new_path, v


def cartesian(sweep):
    sweep_groups = list(find_sweep_groups(sweep))
    keys = [p for p, _ in sweep_groups]
    values = [v for _, v in sweep_groups]

    for combination in product(*values):
        temp_dict = copy.deepcopy(sweep)
        for path, value in zip(keys, combination):
            sub_dict = temp_dict
            for key in path[:-1]:
                sub_dict = sub_dict.setdefault(key, {})
            sub_dict[path[-1].replace(SWEEP_KEY, "")] = value
        yield temp_dict


def dump_configs(configs, filepath):
    filepath = Path(filepath).with_suffix("")
    dir = str(filepath.parent).replace("sweep", "autogen")
    dir = Path("configs/" + dir)
    dir.mkdir(parents=True, exist_ok=True)
    endfile = filepath.name
    for i, config in enumerate(configs):
        config_filepath = f"{dir}/{endfile}_{i}.yaml"
        dump(config, config_filepath)
        yield config_filepath


def generate_configs_from(filepath):
    sweep = load_sweep(filepath)
    configs = [Config(**c) for c in cartesian(sweep)]
    return list(dump_configs(configs, filepath))
