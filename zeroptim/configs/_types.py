from pydantic import BaseModel, validator, root_validator
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from faker import Faker
import random

from zeroptim.supported import (
    __supported_models__,
    __supported_optims__,
    __supported_criterions__,
    __supported_datasets__,
)


class WandbConfig(BaseModel):
    # required wandb parameters
    project_name: str
    mode: Literal["offline", "online"]
    run_name: Optional[str]  # can be auto-generated
    timestamp: Optional[str]  # auto-generated

    @root_validator(pre=True)
    def generate_timestamp(cls, values):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        values["timestamp"] = timestamp
        return values

    @root_validator(pre=True)
    def generate_run_name_if_offline_and_missing(cls, values):
        if not values.get("run_name") and values.get("mode") == "offline":
            fake = Faker()
            randseq = "-".join((fake.word(), fake.word()))
            values["run_name"] = f"{randseq}"
        else:
            values["run_name"] = None
        return values


class ExecutionConfig(BaseModel):
    # optional execution hardware parameters
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32", "float64"] = "float32"
    seed: int = random.randint(0, 2**32 - 1)


class DatasetConfig(BaseModel):
    # required dataset parameters
    name: str
    params: Optional[Dict[str, Any]]
    task: Literal["classification", "regression"]

    @validator("name")
    def validate_dataset_name(cls, v):
        assert v in __supported_datasets__, f"Dataset {v} is not supported"
        return v


class ModelConfig(BaseModel):
    # required model parameters
    model_type: str
    model_hparams: Dict[str, Any]

    @validator("model_type")
    def validate_model_type(cls, v):
        assert v in __supported_models__, f"Model {v} is not supported"
        return v

    class Config:
        # remove Pydantic warning..
        protected_namespaces = ()


class OptimConfig(BaseModel):
    # required optim parameters
    criterion_type: str
    optimizer_type: str
    opt_params: Dict[str, Any]

    # if zero-order optimizer
    epsilon: Optional[float] = None
    sub_optimizer_type: Optional[str] = None

    @validator("criterion_type")
    def validate_criterion_type(cls, v):
        assert v in __supported_criterions__, f"Criterion {v} is not supported"
        return v

    @validator("optimizer_type")
    def validate_optimizer_type(cls, v):
        assert v in __supported_optims__, f"Optimizer {v} is not supported"
        return v

    @root_validator(skip_on_failure=True)
    def check_sub_optimizer(cls, values):
        optimizer_type = values.get("optimizer_type")

        if optimizer_type in ["mezo", "smartes"]:
            sub_optimizer_type = values.get("sub_optimizer_type")
            if not sub_optimizer_type:
                raise ValueError(
                    f"'sub_optimizer_type' must be specified for optimizer_type '{optimizer_type}'"
                )
            if sub_optimizer_type in ["mezo", "smartes"]:
                raise ValueError(
                    f"'sub_optimizer_type' cannot be '{sub_optimizer_type}' for optimizer_type '{optimizer_type}'"
                )

            if sub_optimizer_type and sub_optimizer_type not in __supported_optims__:
                raise ValueError(
                    f"'sub_optimizer_type' {sub_optimizer_type} is not a supported optimizer"
                )

        return values


class Experimental(BaseModel):
    svd: bool = False
    landscape: Literal["batch", "partial", "full"] = "batch"
    n_add_batch: int = 10  # used if landscape == "partial"
    layerwise: bool = False
    frequency: int = 1


class Config(BaseModel):
    wandb: WandbConfig
    env: ExecutionConfig
    dataset: DatasetConfig
    model: ModelConfig
    optim: OptimConfig
    sharpness: Experimental
