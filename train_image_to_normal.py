from pytorch_lightning.cli import LightningCLI

from imagetonormal.models.module import Module
from imagetonormal.data.images import ImageDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DeepSpeedStrategy
import torch


def cli_main():
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="val/loss", save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    cli = LightningCLI(
        Module,
        ImageDataModule,
        trainer_defaults={
            "callbacks": [checkpoint_callback, lr_monitor],
        },
    )


if __name__ == "__main__":
    cli_main()
