import datetime
import os
import sys

import typer
import lightning as L
from loguru import logger

# from sklearn.model_selection import train_test_split

from src import (
    config,
    data_preprocess,
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

    data = data_preprocess.read_data(config.DataConfig.train_data_path)
    question_embeddings = data_preprocess.read_embeddings(
        config.DataConfig.questions_embeddings_path
    )
    concept_embeddings = data_preprocess.read_embeddings(config.DataConfig.concepts_embeddings_path)
    train_data, valid_data = data_preprocess.train_test_split(data, trainer_config)
    logger.info("Data split into train and valid")

    train_loader, valid_loader = data_pytorch.get_data_loaders(
        train_data, valid_data, question_embeddings, concept_embeddings, trainer_config
    )
    logger.info("Data loaders created")

    if trainer_config.kc_model_config.use_lstm:
        kc_model = model.KCLSTMModel(trainer_config.kc_model_config)
    else:
        kc_model = model.KCDecoderAttentionModel(trainer_config.kc_model_config)
    logger.info("Model created")

    logger.info("Starting training")
    trainer.train(kc_model, train_loader, valid_loader, trainer_config)
    logger.info("Training complete")


# Entry point for the CLI app
if __name__ == "__main__":
    app()
