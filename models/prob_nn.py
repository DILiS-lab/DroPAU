import sys, os

sys.path.append(os.path.abspath("../"))
from typing import Optional, Tuple
from .uncertainty_aware_model import (
    UncertaintyAwareModel,
)
from CLUE.src.utils import to_variable
from .utils.data import RegressionDataset
from .utils.metrics import BetaGNLLLoss
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from .utils.utils import remove_checkpoints


class InnerModel(torch.nn.Module):
    def __init__(
        self,
        n_features,
        n_units_per_layer=None,
        dropout_prob=None,
    ):
        super().__init__()
        self.fully_connected_layers = nn.ModuleList()
        self.fully_connected_layers.append(nn.Linear(n_features, n_units_per_layer[0]))
        for i in range(1, len(n_units_per_layer)):
            self.fully_connected_layers.append(
                nn.Linear(n_units_per_layer[i - 1], n_units_per_layer[i])
            )
        self.fully_connected_layers.append(nn.Linear(n_units_per_layer[-1], 2))

        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        for layer in self.fully_connected_layers[:-2]:
            if self.dropout_prob is not None:
                x = self.dropout(x)
            x = torch.relu(layer(x))
        x = torch.relu(self.fully_connected_layers[-2](x))
        x = self.fully_connected_layers[-1](x)

        mean = x[:, 0]
        variance = x[:, 1]
        # enforce positive variance
        variance = torch.exp(variance)

        # print(mean.unsqueeze(1).shape, variance.unsqueeze(1).shape)

        return torch.cat([mean.unsqueeze(1), variance.unsqueeze(1)], dim=1)


class ProbabilisticFeedForwardNetwork(pl.LightningModule, UncertaintyAwareModel):
    def __init__(
        self,
        n_features,
        n_units_per_layer=None,
        dropout_prob=None,
        cuda: bool = False,
        beta_gaussian: bool = False,
    ) -> None:
        if n_units_per_layer is None:
            n_units_per_layer = [1024, 64]

        super().__init__()
        self.cuda = cuda
        self.model = InnerModel(
            n_features=n_features,
            n_units_per_layer=n_units_per_layer,
            dropout_prob=dropout_prob,
        )
        if cuda:
            self.model = self.model.to("cuda")
        if beta_gaussian:
            self.nll_loss = BetaGNLLLoss()
        else:
            self.nll_loss = nn.GaussianNLLLoss()

        self.mse_loss = nn.MSELoss()
        self.mse_only = False

    def set_mode_train(self, train=True):
        if train:
            self.train()
            self.model.train()
        else:
            self.eval()
            self.model.eval()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: Optional[np.ndarray],
        y_eval: Optional[np.ndarray],
        trainer_params: Optional[dict] = None,
        batch_size=32,
        patience=3,
        checkpoint_path: Optional[str] = None,
        adversarial_training: bool = False,
        adversarial_epsilon: float = 0.01,
        num_workers: int = 2,
    ):
        self.adversarial_training = adversarial_training
        self.adversarial_epsilon = adversarial_epsilon
        if trainer_params is None:
            trainer_params = {"progress_bar_refresh_rate": 0, "max_epochs": 100}

        train_dataset = RegressionDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        if (X_eval is not None) and (y_eval is not None):
            val_dataset = RegressionDataset(X_eval, y_eval)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        # Train the model
        monitor = "train_loss" if ((X_eval is None) or (y_eval is None)) else "val_loss"

        early_stop_callback = EarlyStopping(
            monitor=monitor, mode="min", patience=patience
        )
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_path, monitor=monitor, mode="min", save_top_k=1
        )

        progress_bar = TQDMProgressBar(
            refresh_rate=trainer_params["progress_bar_refresh_rate"]
        )
        trainer_params_copy = trainer_params.copy()
        del trainer_params_copy["progress_bar_refresh_rate"]

        # Initialize the Lightning trainer
        trainer = pl.Trainer(
            callbacks=[early_stop_callback, self.checkpoint_callback, progress_bar],
            **trainer_params_copy
        )
        remove_checkpoints(trainer, checkpoint_path)

        trainer.fit(self, train_loader, val_loader)

    def forward(self, x):
        if x.is_cuda:
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("cpu")
        x = self.model(x)
        mean = x[:, 0]
        variance = x[:, 1]
        return mean, variance

    def predict(self, x, grad=False):
        # outputs mean and std!
        self.set_mode_train(train=False)
        self.training = False
        (x,) = to_variable(var=(x,), cuda=self.cuda)
        if grad and not x.requires_grad:
            x.requires_grad = True
        mu, sigma = self.forward(x)
        sigma = sigma.sqrt()

        if grad:
            return mu, sigma
        else:
            return mu.data, sigma.data

    def adversarial_training_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad = True
        loss = self._forward_loss_and_log(x, y, "train_loss")
        loss.backward(retain_graph=True)
        x_grad = torch.sign(
            x.grad.data
        )  # calculate the sign of gradient of the loss func (with respect to input X) (adv)
        x_adversarial = x.data + self.adversarial_epsilon * x_grad
        loss_adv = self._forward_loss_and_log(x_adversarial, y, "train_loss_adv")
        return loss + loss_adv

    def _forward_loss_and_log(self, x, y, log_as: str):
        mean, variance = self.forward(x)

        if self.mse_only:
            result = self.mse_loss(mean, y)
        else:
            result = self.nll_loss(
                input=mean,
                target=y,
                var=variance,
            )
        self.log(log_as, result)
        return result

    def non_adversarial_training_step(self, batch, batch_idx):
        x, y = batch
        return self._forward_loss_and_log(x, y, "train_loss")

    def training_step(self, batch, batch_idx):
        if self.adversarial_training:
            return self.adversarial_training_step(batch, batch_idx)
        else:
            return self.non_adversarial_training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return self._forward_loss_and_log(x, y, "val_loss")

    def predict_target(self, X: np.ndarray) -> np.ndarray:
        is_training = self.training
        self.set_mode_train(train=False)
        with torch.no_grad():
            X = torch.from_numpy(X).float()
            if self.cuda:
                X = X.cuda()
            mean, _ = self.forward(X)
        self.train(is_training)
        return mean.cpu().numpy()

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        is_training = self.training
        self.set_mode_train(train=False)
        with torch.no_grad():
            _, variance = self.forward(torch.from_numpy(X).float())
        self.train(is_training)
        self.model.train(is_training)
        return variance.cpu().numpy()

    def predict_uncertainty_tensor(self, X: torch.Tensor) -> np.ndarray:
        is_training = self.training
        self.set_mode_train(train=False)
        with torch.no_grad():
            _, variance = self.forward(X.float())
        self.set_mode_train(train=is_training)
        return variance

    def predict_target_and_uncertainty(self, X: np.ndarray) -> np.ndarray:
        is_training = self.training
        self.set_mode_train(train=False)
        with torch.no_grad():
            mean, variance = self.forward(torch.from_numpy(X).float())
        self.set_mode_train(train=is_training)
        return mean.numpy(), variance.numpy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
