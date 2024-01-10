from torch.utils.data import DataLoader
from typing import Dict, Any

from zeroptim.configs._types import Config
from zeroptim.supported import __supported_datasets__
from zeroptim.dataset.vision import VisionDataset

class DataLoaderFactory:

    @staticmethod
    def init_loader(config: Config) -> DataLoader:
        dataset_name: str = config.dataset.name
        dataset_params: Dict[str, Any] = config.dataset.dataset_params or {}
        return VisionDataset.make_loader(dataset_name, **dataset_params)
