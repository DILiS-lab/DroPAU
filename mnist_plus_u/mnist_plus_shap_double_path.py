import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import torch
from tqdm import tqdm

from utils_dataset import get_loaders
from utils_evaluation import Localization, plot_localization, extend_mask
from utils_model import SwitchableDoublePathCNNModule

device = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
torch.set_float32_matmul_precision("medium")

output_dir = "./combined_mnist_dataset"
num_samples = 500000

train_dataset, val_dataset, test_dataset = get_loaders(
    output_dir, num_samples, get_sets="splits"
)

l_model = SwitchableDoublePathCNNModule.load_from_checkpoint(
    "checkpoints_two_models/best_model_1736941275.ckpt"
)
mean_model = l_model.mean_model.to(device)
var_model = l_model.variance_model.to(device)


# Define a PyTorch model wrapper for SHAP
class SHAPMeanVarianceModel(torch.nn.Module):
    def __init__(self, mean_model, var_model):
        super(SHAPMeanVarianceModel, self).__init__()
        self.mean_model = mean_model
        self.var_model = var_model

    def forward(self, inputs):
        means = self.mean_model(inputs)
        log_vars = self.var_model(inputs)
        variances = torch.exp(log_vars)  # Convert log variance to variance
        return torch.cat([means, variances], dim=1)  # Concatenate mean and variance


shap_model = SHAPMeanVarianceModel(mean_model, var_model)

np.random.seed(0)
samples = np.random.choice(list(range(len(val_dataset))), 500)
background = torch.cat([val_dataset[i][0].unsqueeze(0) for i in samples])
print(background.shape)


localization = Localization()
mean_model.eval()
var_model.eval()
shap_model.eval()

for i in tqdm(range(len(test_dataset))):  
    image, label, uc, mean_mask, var_mask = test_dataset[
        i
    ]  # Get a sample from the dataset
    # print(image.unsqueeze(0).shape)

    explainer = shap.DeepExplainer(shap_model, background.to(device))
    shap_values = explainer.shap_values(image.unsqueeze(0).to(device))

    mean_heatmap = torch.Tensor(shap_values[0][0][0])
    var_heatmap = torch.Tensor(shap_values[1][0][0])

    localization.add_sample(
        mean_heatmap.cpu().detach(),
        extend_mask(mean_mask, 2).cpu().detach(),
        var_heatmap.cpu().detach(),
        extend_mask(var_mask, 2).cpu().detach(),
        uc.detach(),
    )


localization.save(file_name="mnist_plus_shap_500_double_path_extended2")

localization_result = localization.calculate_acc_matrices()

plot_localization(
    localization_result, save_path="mnist_plus_shap_500_double_path_extended2.png"
)
