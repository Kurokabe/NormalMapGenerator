import torch
from imagetonormal.models.unet import UNet
import pytorch_lightning as pl
from torch.nn import MSELoss, L1Loss
import torchvision


class Module(pl.LightningModule):
    def __init__(self, d: int = 64):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(d)
        self.mse = MSELoss()
        self.l1 = L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse = self.mse(y_hat, y)
        l1 = self.l1(y_hat, y)
        loss = mse + l1
        self.log("train/mse", mse, sync_dist=True)
        self.log("train/l1", l1, sync_dist=True)
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.sample_y = []
        self.sample_y_hat = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.sample_y.append(y)
        self.sample_y_hat.append(y_hat)
        mse = self.mse(y_hat, y)
        l1 = self.l1(y_hat, y)
        loss = mse + l1
        self.log("val/mse", mse, sync_dist=True)
        self.log("val/l1", l1, sync_dist=True)
        self.log("val/loss", loss, sync_dist=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.sample_y = torch.cat(self.sample_y)
        self.sample_y_hat = torch.cat(self.sample_y_hat)
        self.sample_y = self.sample_y[:16]
        self.sample_y_hat = self.sample_y_hat[:16]

        grid_real = torchvision.utils.make_grid(
            self.sample_y, nrow=4, normalize=True, value_range=(0, 1)
        )
        grid_fake = torchvision.utils.make_grid(
            self.sample_y_hat, nrow=4, normalize=True, value_range=(0, 1)
        )

        self.logger.experiment.add_image("images/y", grid_real, self.current_epoch)
        self.logger.experiment.add_image("images/y_hat", grid_fake, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
