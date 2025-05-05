import time
from utils_dataset import  get_loaders
from utils_model import SwitchableDoublePathCNNModule
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

output_dir = "./combined_mnist_dataset"
num_samples = 500000

train_dataloader, val_dataloader, test_dataloader = get_loaders(output_dir, num_samples)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_float32_matmul_precision('medium')

# Callbacks
filename = f"best_model_{int(time.time())}"
checkpoint_callback = ModelCheckpoint(
    monitor="val_gaussian_nll",
    dirpath="./checkpoints_two_models",
    filename=filename,
    save_top_k=1,
    mode="min"
)

early_stopping_callback = EarlyStopping(
    monitor="val_gaussian_nll",
    patience=10,
    mode="min"
)


switch_epoch = 16

# Trainer
trainer = pl.Trainer(
    min_epochs=switch_epoch+1,
    max_epochs=35,
    callbacks=[checkpoint_callback, early_stopping_callback],
    accelerator=device,
    devices=1
)


# Training
l_model = SwitchableDoublePathCNNModule(switch_epoch=switch_epoch, lr=0.001, weight_decay=1e-3, clip_value=1.0)
trainer.fit(l_model, train_dataloader, val_dataloader)

print("Best Model Path:", checkpoint_callback.best_model_path)