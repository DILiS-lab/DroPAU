import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from utils_dataset import get_loaders
from utils_evaluation import Localization, plot_localization, extend_mask
from utils_model import SwitchableDoublePathCNNModule


device = (
    "cuda:0"
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


# Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(self):
        # Average the gradients spatially
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted sum of activations
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()

        heatmap = heatmap + heatmap.min()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return heatmap


# Instantiate the Grad-CAM object for the shared CNN layers
# We use the last convolutional layer in the shared CNN network
mean_target_layer = mean_model.cnn[
    3
]  # This is the last convolutional layer before flattening
var_target_layer = var_model.cnn[
    3
]  # This is the last convolutional layer before flattening

mean_gradcam = GradCAM(mean_model, mean_target_layer)
var_gradcam = GradCAM(var_model, var_target_layer)

# samples = np.random.choice(range(len(dataset) + 1), size=10, replace=False)
# Test Grad-CAM on a sample image

localization = Localization()

mean_model.eval()
var_model.eval()
for i in [1520]:  # tqdm(range(len(test_dataset))):  # Loop over 10 samples
    image, label, uc, mean_mask, var_mask = test_dataset[
        i
    ]  # Get a sample from the dataset
    image = image.unsqueeze(0)  # Add batch dimension

    # Forward pass
    # print(model(image.to(device)).unsqueeze(0).shape)
    mean_output = mean_model(image.to(device))
    log_var_output = var_model(image.to(device))

    # Grad-CAM for mean
    mean_output.backward()  # Backpropagate for mean
    mean_heatmap = mean_gradcam.generate_heatmap()

    # Reset gradients after mean backward pass
    mean_model.zero_grad()

    # Grad-CAM for variance
    log_var_output.backward()  # Backpropagate for variance
    var_heatmap = var_gradcam.generate_heatmap()
    # Reset gradients after mean backward pass
    var_model.zero_grad()

    localization.add_sample(
        mean_heatmap.cpu().detach(),
        extend_mask(mean_mask, 2).cpu().detach(),
        var_heatmap.cpu().detach(),
        extend_mask(var_mask, 2).cpu().detach(),
        uc.detach(),
    )
    del mean_output
    del log_var_output
    torch.cuda.empty_cache()



localization.save(file_name="mnist_plus_gradcam_double_path_extended2")
localization_result = localization.calculate_acc_matrices()

plot_localization(
    localization_result, save_path="mnist_plus_gradcam_double_path_extended2.png"
)
