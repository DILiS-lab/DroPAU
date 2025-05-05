import torch
from tqdm import tqdm
import numpy as np

from utils_dataset import get_loaders
from utils_evaluation import Localization, plot_localization, extend_mask
from utils_model import SwitchableDoublePathCNNModule
from captum.attr import IntegratedGradients

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
torch.set_float32_matmul_precision("medium")

output_dir = "./combined_mnist_dataset"
num_samples = 500000  # Number of combined images to create

train_dataset, val_dataset, test_dataset = get_loaders(
    output_dir, num_samples, get_sets="splits"
)


l_model = SwitchableDoublePathCNNModule.load_from_checkpoint(
    "checkpoints_two_models/best_model_1736941275.ckpt"
)
mean_model = l_model.mean_model.to(device)
var_model = l_model.variance_model.to(device)


localization = Localization()
mean_model.eval()
var_model.eval()

for i in tqdm(range(len(test_dataset))):  
    image, label, uc, mean_mask, var_mask = test_dataset[
        i
    ]  # Get a sample from the dataset
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    ig_mean = IntegratedGradients(mean_model)
    ig_var = IntegratedGradients(var_model)

    mean_heatmap, _ = ig_mean.attribute(
        image,
        target=0,
        method="gausslegendre",
        return_convergence_delta=True,
    )
    mean_heatmap = mean_heatmap.squeeze().squeeze()
    mean_model.zero_grad()
    var_heatmap, _ = ig_var.attribute(
        image,
        target=0,
        method="gausslegendre",
        return_convergence_delta=True,
    )
    var_heatmap = var_heatmap.squeeze().squeeze()
    var_model.zero_grad()

    localization.add_sample(
        mean_heatmap.cpu().detach(),
        extend_mask(mean_mask, 2).cpu().detach(),
        var_heatmap.cpu().detach(),
        extend_mask(var_mask, 2).cpu().detach(),
        uc.detach(),
    )


localization.save("mnist_plus_ig_double_path_extended2")

localization_result = localization.calculate_acc_matrices()
plot_localization(
    localization_result, save_path="mnist_plus_ig_double_path_extended2_nomean.png"
)
