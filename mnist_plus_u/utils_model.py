import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import mean_squared_error


class MeanVarianceCNN(nn.Module):
    def __init__(self, focus=None):
        super(MeanVarianceCNN, self).__init__()
        self.focus = focus
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 32 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.cnn(x)
        # x = x.reshape(x.size(0), -1)  # Flatten
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class MeanVarianceSwitchableCNN(nn.Module):
    def __init__(self, focus=None):
        super(MeanVarianceSwitchableCNN, self).__init__()
        self.focus = focus
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 32 * 64, 128), nn.ReLU(), nn.Dropout(0.3)
        )

        self.fc_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.fc_log_var = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)

        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        if self.focus is None:
            return mean, log_var
        elif self.focus == "mean":
            return mean
        else:
            return log_var


# Define the Lightning Module
class SwitchableCNNModule(pl.LightningModule):
    def __init__(self, switch_epoch=15, lr=0.001, weight_decay=1e-3, clip_value=5.0):
        super(SwitchableCNNModule, self).__init__()
        self.model = MeanVarianceSwitchableCNN()
        self.switch_epoch = switch_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip_value = clip_value
        self.automatic_optimization = True
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels, gt_uncertainty, _, _ = batch
        means, log_vars = self.model(images)
        variances = torch.exp(log_vars)

        # Determine the loss function
        current_epoch = self.current_epoch
        if current_epoch < self.switch_epoch:
            loss_function = nn.MSELoss()
            loss = loss_function(means, labels)
        else:
            loss_function = nn.GaussianNLLLoss()
            loss = loss_function(means, labels, variances)

        # Log training metrics
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        # Log Gaussian NLL as a metric regardless of the loss used
        # variances = torch.exp(log_vars)
        gaussian_nll = nn.GaussianNLLLoss()(means, labels, variances)
        self.log(
            "train_mse",
            mean_squared_error(means, labels),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train_gaussian_nll",
            gaussian_nll,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, gt_uncertainty, _, _ = batch
        means, log_vars = self.model(images)
        variances = torch.exp(log_vars)

        # Compute Gaussian NLL
        gaussian_nll = nn.GaussianNLLLoss()(means, labels, variances)
        self.log(
            "val_mse",
            mean_squared_error(means, labels),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "val_gaussian_nll",
            gaussian_nll,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return gaussian_nll

    def test_step(self, batch, batch_idx):
        images, labels, gt_uncertainty, _, _ = batch
        means, log_vars = self.model(images)
        variances = torch.exp(log_vars)

        # Compute Gaussian NLL
        gaussian_nll = nn.GaussianNLLLoss()(means, labels, variances)
        self.log(
            "test_gaussian_nll",
            gaussian_nll,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return gaussian_nll

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, labels, gt_uncertainty, mean_mask, uncertainty_mask = batch
        means, log_vars = self.model(images)
        return means, log_vars, labels, gt_uncertainty, mean_mask, uncertainty_mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)
        optimizer.step(closure=optimizer_closure)


class SwitchableDoublePathCNNModule(pl.LightningModule):
    def __init__(self, switch_epoch=15, lr=0.001, weight_decay=1e-3, clip_value=5.0):
        super(SwitchableDoublePathCNNModule, self).__init__()
        self.mean_model = MeanVarianceCNN()
        self.variance_model = MeanVarianceCNN()
        self.switch_epoch = switch_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip_value = clip_value
        self.automatic_optimization = True
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        images, labels, gt_uncertainty, _, _ = batch
        means = self.mean_model(images)
        log_vars = self.variance_model(images)
        variances = torch.exp(log_vars)

        # Determine the loss function
        current_epoch = self.current_epoch
        if current_epoch < self.switch_epoch:
            loss_function = nn.MSELoss()
            loss = loss_function(means, labels)
        else:
            loss_function = nn.GaussianNLLLoss()
            loss = loss_function(means, labels, variances)

        # Log training metrics
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        # Log Gaussian NLL as a metric regardless of the loss used
        # variances = torch.exp(log_vars)
        gaussian_nll = nn.GaussianNLLLoss()(means, labels, variances)
        self.log(
            "train_mse",
            mean_squared_error(means, labels),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train_gaussian_nll",
            gaussian_nll,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, gt_uncertainty, _, _ = batch
        means = self.mean_model(images)
        log_vars = self.variance_model(images)
        variances = torch.exp(log_vars)
        # print(means.shape, variances.shape, labels.shape)
        # Compute Gaussian NLL
        gaussian_nll = nn.GaussianNLLLoss()(means, labels, variances)
        self.log(
            "val_mse",
            mean_squared_error(means, labels),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "val_gaussian_nll",
            gaussian_nll,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return gaussian_nll

    def test_step(self, batch, batch_idx):
        images, labels, gt_uncertainty, _, _ = batch
        means = self.mean_model(images)
        log_vars = self.variance_model(images)
        variances = torch.exp(log_vars)
        # Compute Gaussian NLL
        gaussian_nll = nn.GaussianNLLLoss()(means, labels, variances)
        self.log(
            "test_gaussian_nll",
            gaussian_nll,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return gaussian_nll

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images, labels, gt_uncertainty, mean_mask, uncertainty_mask = batch
        means = self.mean_model(images)
        log_vars = self.variance_model(images)
        return means, log_vars, labels, gt_uncertainty, mean_mask, uncertainty_mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)
        optimizer.step(closure=optimizer_closure)


class LogSquaredResisuals(nn.Module):
    def __init__(self):
        super(LogSquaredResisuals, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, y_pred, y_true):
        return torch.log(self.mse(y_pred, y_true) + 1e-6)
