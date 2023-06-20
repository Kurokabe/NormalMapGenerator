import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch.nn import L1Loss, MSELoss

from imagetonormal.models.discriminator import NLayerDiscriminator, weights_init
from imagetonormal.models.unet import UNet


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


class Module(pl.LightningModule):
    def __init__(self, d: int = 64, disc_loss="hinge"):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(d)
        self.mse = MSELoss()
        self.l1 = L1Loss()

        self.discriminator = NLayerDiscriminator(
            input_nc=3,
            n_layers=3,
            use_actnorm=False,
            ndf=64,
        ).apply(weights_init)

        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight  # * self.discriminator_weight
        return d_weight

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        y_hat = self(x)

        rec_loss = torch.mean(torch.abs(y.contiguous() - y_hat.contiguous()))

        if optimizer_idx == 0:
            self.discriminator.eval()
            self.model.train()
            logits_fake = self.discriminator(y_hat.contiguous())
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(
                    rec_loss, g_loss, last_layer=self.model.last_conv.weight
                )
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            loss = rec_loss + d_weight * g_loss

            self.log("train/rec_loss", rec_loss, sync_dist=True)
            self.log("train/g_loss", g_loss, sync_dist=True)
            self.log("train/loss", loss, sync_dist=True)
        elif optimizer_idx == 1:
            self.model.eval()
            self.discriminator.train()
            logits_real = self.discriminator(y.contiguous().detach())
            logits_fake = self.discriminator(y_hat.contiguous().detach())

            d_loss = self.disc_loss(logits_real, logits_fake)

            loss = d_loss

            # self.log("train/mse", mse, sync_dist=True)
            self.log("train/disc_loss", d_loss, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.sample_y = []
        self.sample_y_hat = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.sample_y.append(y)
        self.sample_y_hat.append(y_hat)

        rec_loss = torch.mean(torch.abs(y.contiguous() - y_hat.contiguous()))

        logits_fake = self.discriminator(y_hat.contiguous())
        g_loss = -torch.mean(logits_fake)

        try:
            d_weight = self.calculate_adaptive_weight(
                rec_loss, g_loss, last_layer=self.model.last_conv.weight
            )
        except RuntimeError:
            assert not self.training
            d_weight = torch.tensor(0.0)

        loss = rec_loss + d_weight * g_loss

        self.log("val/rec_loss", rec_loss, sync_dist=True)
        self.log("val/g_loss", g_loss, sync_dist=True)
        self.log("val/loss", loss, sync_dist=True)

        logits_real = self.discriminator(y.contiguous().detach())

        d_loss = self.disc_loss(logits_real, logits_fake)

        loss = d_loss

        # self.log("val/mse", mse, sync_dist=True)
        self.log("val/disc_loss", d_loss, sync_dist=True)

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
        opt_unet = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.5, 0.9),
        )

        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.9),
            # eps=1e-6,
        )
        return [opt_unet, opt_disc], []
