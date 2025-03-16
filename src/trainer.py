import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

from src import config


class LightningModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        loss_fn: nn.Module,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.accuracy = nn.ModuleDict(
            {
                "train_acc": torchmetrics.Accuracy(task="binary"),
                "valid_acc": torchmetrics.Accuracy(task="binary"),
            }
        )
        self.auroc = nn.ModuleDict(
            {
                "train_auroc": torchmetrics.AUROC(task="binary"),
                "valid_auroc": torchmetrics.AUROC(task="binary"),
            }
        )

    def common_step(self, x: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        out = self.model(x)

        loss_all = self.loss_fn(out, x["response"][..., None].float()).squeeze(dim=-1)
        loss = loss_all[x["attention_mask"]].mean()
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True)

        preds = out[x["attention_mask"]].sigmoid().squeeze()
        ys = x["response"][x["attention_mask"]]
        if len(preds) > 0:
            self.accuracy[f"{prefix}_acc"](preds, ys)
            self.auroc[f"{prefix}_auroc"](preds, ys)
            self.log(
                f"{prefix}_accuracy", self.accuracy[f"{prefix}_acc"], on_step=True, on_epoch=True
            )
            self.log(f"{prefix}_auroc", self.auroc[f"{prefix}_auroc"], on_step=True, on_epoch=True)

        return loss

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self.common_step(batch, prefix="train")

        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ):
        _ = self.common_step(batch, prefix="valid")

        # if batch_idx == 0:
        #     self.log_examples(batch)

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]


def train(
    kc_model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    trainer_config: config.TrainerConfig,
):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    lightning_module = LightningModule(kc_model, trainer_config.learning_rate, loss_fn)
    logger = WandbLogger()
    trainer = L.Trainer(
        max_epochs=3 if trainer_config.is_local else trainer_config.num_epochs,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        gradient_clip_val=1.0,
        precision=32,
        num_sanity_val_steps=0,
        logger=logger,
        enable_progress_bar=trainer_config.is_local,
        log_every_n_steps=100,
        # limit_train_batches=20 if trainer_config.is_local else 1.0,
        # limit_val_batches=3 if trainer_config.is_local else 1.0,
    )
    trainer.fit(lightning_module, train_loader, valid_loader)
