import hashlib
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.functional import mean_squared_error
from torchvision import datasets, transforms
from .utils_model import SwitchableCNNModule
from .utils_dataset import CombinedMNISTDataset
import sys
sys.path.append("..")
from utils.experiment_utils import clue_explain_images
from .utils_evaluation import Localization

output_dir = "./combined_mnist_dataset"
num_samples = 500000  # Number of combined images to create

# Create PyTorch Dataset
transform = transforms.Compose(
    [transforms.ToTensor()]
)  # , v2.RandomRotation(degrees=15)])
dataset = CombinedMNISTDataset(data_dir=output_dir, transform=transform)


generator1 = torch.Generator().manual_seed(0)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [int(0.7 * num_samples), int(0.1 * num_samples), int(0.2 * num_samples)],
    generator=generator1,
)


train_dataloader = DataLoader(
    train_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=4
)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
torch.set_float32_matmul_precision("medium")


model = SwitchableCNNModule.load_from_checkpoint(
    "checkpoints/best_model_1736245990.ckpt"
).model


SKIP_VAE_TRAINING = True  # already trained
CPU = True
pnn = model
pnn = pnn.to(torch.float32)
if CPU:
    pnn = pnn.cpu()
identifier = "combined_mnist"
print("reducing dataset size for clue as it is too slow")
trainset = torch.utils.data.Subset(train_dataset, range(2 * len(train_dataset) // 5))
valset = torch.utils.data.Subset(val_dataset, range(2 * len(val_dataset) // 5))
testset = torch.utils.data.Subset(test_dataset, range(2 * len(test_dataset) // 5))
ind_instances_to_explain = range(len(testset))

del train_dataset, val_dataset, test_dataset

clue_explain_images(
    pnn,
    trainset,
    valset,
    testset,
    ind_instances_to_explain,
    identifier,
    save_dir="combined_mnist_vae",
    save=True,
    sort=False,
    skip_vae_training=SKIP_VAE_TRAINING,
    cpu=CPU,
)
data = np.load(
    "combined_mnist_vae/importances/CLUE_importances_combined_mnist.npy",
    allow_pickle=True,
)
variance_explanations = np.abs(data.item()["importances_directed"])
localization = Localization()

for i in range(len(testset)):
    image, label, uc, mean_mask, var_mask = testset[i]  # Get a sample from the dataset
    explanation = variance_explanations[i]
    mean_heatmap = np.ones_like(explanation)
    var_heatmap = explanation
    localization.add_sample(
        torch.Tensor(mean_heatmap), mean_mask, torch.Tensor(var_heatmap), var_mask
    )


localization_result = localization.calculate_acc_matrices()

# Plotting the heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# IoU heatmap
sns.heatmap(
    localization_result["iou_matrix"],
    annot=localization_result["iou_annot"],
    cmap="Blues",
    xticklabels=["Mean Heatmap", "Variance Heatmap"],
    yticklabels=["Mean Mask", "Variance Mask"],
    ax=axes[0],
    fmt="",
)
axes[0].set_title("IoU Heatmap (Avg. ± Std.)")

# Mass accuracy heatmap
sns.heatmap(
    localization_result["mass_acc_matrix"],
    annot=localization_result["mass_acc_annot"],
    cmap="Blues",
    xticklabels=["Mean Heatmap", "Variance Heatmap"],
    yticklabels=["Mean Mask", "Variance Mask"],
    ax=axes[1],
    fmt="",
)
axes[1].set_title("Mass Accuracy Heatmap (Avg. ± Std.)")

# Rank accuracy heatmap
sns.heatmap(
    localization_result["rank_acc_matrix"],
    annot=localization_result["rank_acc_annot"],
    cmap="Blues",
    xticklabels=["Mean Heatmap", "Variance Heatmap"],
    yticklabels=["Mean Mask", "Variance Mask"],
    ax=axes[2],
    fmt="",
)
axes[2].set_title("Rank Accuracy Heatmap (Avg. ± Std.)")

plt.tight_layout()
plt.savefig("localization_heatmaps_clue.png")

print("done")
