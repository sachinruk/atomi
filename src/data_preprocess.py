import json
import pathlib

from loguru import logger
import pandas as pd
import torch

from src import config


def read_data(data_path: pathlib.Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    for column in config.DataConfig.columns:
        if column in data.columns:
            data[column] = data[column].map(lambda x: [int(val) for val in x.split(",")])

    logger.info(f"Data loaded from {data_path} with shape {data.shape}")
    return data


def read_embeddings(json_path: pathlib.Path) -> torch.Tensor:
    with open(json_path, "r") as f:
        embeddings = json.load(f)

    torch_embeddings = torch.Tensor([embeddings[str(i)] for i in range(len(embeddings))])

    logger.info(f"Embeddings loaded from {json_path} with shape {torch_embeddings.shape}")
    return torch_embeddings


def train_test_split(
    data: pd.DataFrame, trainer_config: config.TrainerConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = data["fold"] == trainer_config.test_fold
    test_data = data[idx]
    train_data = data[~idx]
    return train_data, test_data
