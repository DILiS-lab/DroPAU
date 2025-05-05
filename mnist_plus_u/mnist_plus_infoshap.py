import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import shap
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms

from utils_dataset import CombinedMNISTDatasetInfoShap, get_loaders
from utils_evaluation import Localization, plot_localization, extend_mask
from utils_model import LogSquaredResisuals, SwitchableCNNModule
import os
from tqdm import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
torch.set_float32_matmul_precision("medium")

output_dir = "./combined_mnist_dataset"
num_samples = 500000

is_dataset_name = "infoshap.dataset_model_1736245990.csv"
if is_dataset_name not in os.listdir("."):

    dataset = get_loaders(output_dir, num_samples, get_sets="full")

    l_model = SwitchableCNNModule.load_from_checkpoint(
        "checkpoints/best_model_1736245990.ckpt"
    )

    full_dataloader = DataLoader(
        dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=8
    )

    trainer = pl.Trainer(accelerator=device, devices=1)
    var_result = trainer.predict(l_model, full_dataloader)

    means = torch.cat([batch[0] for batch in var_result])
    labels = torch.cat([batch[2] for batch in var_result])

    log_sqrd_res = LogSquaredResisuals()
    new_label = log_sqrd_res(means, labels)

    new_dataset_file = dataset.data.copy().assign(new_label=new_label.numpy())

    new_dataset_file.to_csv(is_dataset_name, index=False)

transform = transforms.Compose([transforms.ToTensor()])
infoshap_dataset = CombinedMNISTDatasetInfoShap(
    data_dir=output_dir, metadata=pd.read_csv(is_dataset_name), transform=transform
)
generator1 = torch.Generator().manual_seed(0)
infoshap_train_dataset, infoshap_val_dataset, infoshap_test_dataset = (
    torch.utils.data.random_split(
        infoshap_dataset,
        [int(0.7 * num_samples), int(0.1 * num_samples), int(0.2 * num_samples)],
        generator=generator1,
    )
)

infoshap_train_dataloader = DataLoader(
    infoshap_train_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=8
)
infoshap_val_dataloader = DataLoader(
    infoshap_val_dataset, batch_size=256, shuffle=False, num_workers=4
)
infoshap_test_dataloader = DataLoader(
    infoshap_test_dataset, batch_size=64, shuffle=False, num_workers=4
)

is_filename = "is_best_model_1736245990"
# check if is_filename exists
if f"{is_filename}.ckpt" not in os.listdir("./infoshap_checkpoints"):

    is_checkpoint_callback = ModelCheckpoint(
        monitor="val_mse",
        dirpath="./infoshap_checkpoints",
        filename=is_filename,
        save_top_k=1,
        mode="min",
    )

    is_early_stopping_callback = EarlyStopping(
        monitor="val_mse", patience=10, mode="min"
    )

    switch_epoch = 36

    # Trainer
    is_trainer = pl.Trainer(
        max_epochs=35,
        callbacks=[is_checkpoint_callback, is_early_stopping_callback],
        accelerator=device,
        devices=1,
    )

    # Training

    is_l_model = SwitchableCNNModule(
        switch_epoch=switch_epoch, lr=0.001, weight_decay=1e-3, clip_value=1.0
    )
    is_trainer.fit(l_model, infoshap_train_dataloader, infoshap_val_dataloader)

    print("Best Model Path:", is_checkpoint_callback.best_model_path)

is_model_l = SwitchableCNNModule.load_from_checkpoint(
    "infoshap_checkpoints/is_best_model_1736245990.ckpt"
)
is_model = is_model_l.model

np.random.seed(0)
samples = np.random.choice(list(range(len(infoshap_val_dataset))), 500)
background = torch.cat([infoshap_val_dataset[i][0].unsqueeze(0) for i in samples])


localization = Localization()
is_model.eval()
is_model.focus = "mean"

for i in tqdm(range(len(infoshap_test_dataset))): 
    image, label, uc, mean_mask, var_mask = infoshap_test_dataset[
        i
    ]  # Get a sample from the dataset
    # print(image.unsqueeze(0).shape)

    explainer = shap.DeepExplainer(is_model, background.to(device))
    shap_values = explainer.shap_values(image.unsqueeze(0).to(device))
    # print(shap_values.shape)
    # plt.imshow(shap_values[0][0], cmap="bwr")
    # plt.show()
    var_heatmap = torch.Tensor(shap_values[0][0])
    # print(variance_heatmap.shape)
    mean_heatmap = torch.zeros_like(var_heatmap)
    # print(mean_heatmap.sum())

    localization.add_sample(
        mean_heatmap.cpu().detach(),
        extend_mask(mean_mask, 2).cpu().detach(),
        var_heatmap.cpu().detach(),
        extend_mask(var_mask, 2).cpu().detach(),
        uc.detach(),
    )

localization.save(file_name="mnist_plus_infoshap_500_extended2")


localization_result = localization.calculate_acc_matrices()

plot_localization(
    localization_result, save_path="mnist_plus_infoshap_500_extended2.png"
)
