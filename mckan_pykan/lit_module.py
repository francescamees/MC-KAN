import pytorch_lightning as pl
import torch
from .config import MCKANPyKANConfig, OPTIMIZERS
from .model import MCKANPyKAN


class LitMCKANPyKAN(pl.LightningModule):
    def __init__(self, config: MCKANPyKANConfig):
        super().__init__()
        self.config = config
        self.model = MCKANPyKAN(config)
        self.loss_fn = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Z, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Z, batch_Phis)
            losses = [self.loss_fn(pred, y) for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))
        else:
            batch_X, batch_Z, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Z, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2), dim=1) / batch_N) / batch_X.shape[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Z, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Z, batch_Phis)
            losses = [self.loss_fn(pred, y) for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))
        else:
            batch_X, batch_Z, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Z, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2), dim=1) / batch_N) / batch_X.shape[0]
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        if self.config.dataloader_type == 'iterative':
            batch_X, batch_Z, batch_Phis, batch_ys = batch
            preds = self.model(batch_X, batch_Z, batch_Phis)
            losses = [self.loss_fn(pred, y) for pred, y in zip(preds, batch_ys)]
            loss = torch.mean(torch.stack(losses))
        else:
            batch_X, batch_Z, batch_Phi, batch_y, batch_N = batch
            pred = self.model(batch_X, batch_Z, batch_Phi)
            loss = torch.sum(torch.sum(((pred - batch_y) ** 2), dim=1) / batch_N) / batch_X.shape[0]
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.config.training.optimizer](
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        return optimizer
