import datetime
import os
import sys

import typer
import lightning as L
from loguru import logger

# from sklearn.model_selection import train_test_split

from src import (
    config,
    data_pytorch,
    model,
    trainer,
    wandb_utils,
)

app = typer.Typer()


@app.command()
def train(trainer_config_json: str = "{}", training_date: str = "") -> None:
    trainer_config = config.TrainerConfig.model_validate_json(trainer_config_json)
    L.seed_everything(trainer_config.seed)

    if training_date == "":
        now = datetime.datetime.now()
        training_date = now.strftime("%Y%m%d-%H%M%S")
    _ = wandb_utils.get_wandb_logger(trainer_config, training_date)
    logger.info("Initialized wandb logger")


# Entry point for the CLI app
if __name__ == "__main__":
    app()
