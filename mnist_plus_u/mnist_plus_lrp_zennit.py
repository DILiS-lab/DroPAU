import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from zennit.attribution import Gradient
from zennit.composites import NameMapComposite, SpecialFirstLayerMapComposite
from zennit.image import imgify
from zennit.rules import AlphaBeta, Epsilon, Norm, Pass, ZBox

from CLRP.clrp_lib import *
from CLRP.run import clrp, visualize
from utils_dataset import get_loaders
from utils_evaluation import Localization, plot_localization, extend_mask
from utils_model import SwitchableCNNModule

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

l_model = SwitchableCNNModule.load_from_checkpoint(
    "checkpoints/best_model_1736245990.ckpt"
)
model = l_model.model.to(device)

# might need to replace first layer with something else like this correction layer
name_map = [
    (["cnn.0"], ZBox(low=0, high=1, zero_params="bias")),  # Conv2d
    (["cnn.1"], Pass()),  # ReLU
    (["cnn.2"], AlphaBeta(alpha=1, beta=0, zero_params="bias")),  # Conv2d
    (["cnn.3"], Pass()),  # ReLU
    (["cnn.4"], Norm()),  # MaxPool2d
    (["fc.0"], Epsilon(zero_params="bias")),  # Linear
    (["fc.1"], Pass()),  # ReLU
    (["fc.2"], Pass()),  # Dropout
    (["fc_mean.0"], Epsilon(zero_params="bias")),  # Linear
    (["fc_mean.1"], Pass()),  # ReLU
    (["fc_mean.2"], Epsilon(zero_params="bias")),  # Linear
    (["fc_mean.3"], Pass()),  # ReLU
    (["fc_mean.4"], Epsilon(zero_params="bias")),  # Linear
    (["fc_log_var.0"], Epsilon(zero_params="bias")),  # Linear
    (["fc_log_var.1"], Pass()),  # ReLU
    (["fc_log_var.2"], Epsilon(zero_params="bias")),  # Linear
    (["fc_log_var.3"], Pass()),  # ReLU
    (["fc_log_var.4"], Epsilon(zero_params="bias")),  # Linear
]


def get_lrp_explanation(model, img, name_map, exp_target="mean"):
    model.focus = exp_target
    # create composite (the zennit way of specifying the rules)
    composite = NameMapComposite(
        name_map=name_map,
    )

    # choose a target for the attribution by multiplying the output with the identity matrix
    target_base = model(img)
    # print(target_base)
    # appying ReLU if the target is variance
    target = target_base if exp_target == "mean" else target_base.exp()

    # create the attributor - zennit explainer
    with Gradient(model=model, composite=composite) as attributor:
        _, attribution = attributor(img, target)

    # sum over the channels to get the pixel-wise relevance
    relevance = attribution.sum(1)

    # create an image of the visualize attribution
    img = imgify(relevance.cpu(), symmetric=True, cmap="bwr")

    # convert image to RGB mode if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")
    model.focus = None
    # print(relevance.shape)
    return relevance.squeeze()


localization = Localization()

model.eval()
for i in tqdm(range(len(test_dataset))): 
    image, label, uc, mean_mask, var_mask = test_dataset[
        i
    ]  # Get a sample from the dataset
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    # if i == 0:
    #     print(image.min(), image.max())
    # print(image.shape)
    mean_heatmap = get_lrp_explanation(model, image, name_map, exp_target="mean")
    # display(mean_heatmap)
    # Reset gradients after mean backward pass
    model.zero_grad()

    # Grad-CAM for variance
    var_heatmap = get_lrp_explanation(model, image, name_map, exp_target="variance")
    # print(var_heatmap.shape)

    localization.add_sample(
        mean_heatmap.cpu().detach(),
        extend_mask(mean_mask, 2).cpu().detach(),
        var_heatmap.cpu().detach(),
        extend_mask(var_mask, 2).cpu().detach(),
        uc.detach(),
    )

    model.zero_grad()
    # display(var_heatmap)
    # Visualize the heatmaps
    # print("Variance Prediction Heatmap:")
    # visualize_heatmap(image.detach(), mean_heatmap.detach(), var_heatmap.detach(), "lrp")

localization.save(file_name="mnist_plus_lrp_zennit_a1_b0_extended2_zero_bias")

localization_result = localization.calculate_acc_matrices()

plot_localization(
    localization_result, save_path="mnist_plus_lrp_zennit_a1_b0_extended2_zero_bias.png"
)
