import argparse
import os

import torch
import torch.optim as optim
from mivolo.data.dataset.age_gender_dataset import AgeGenderDataset
from mivolo.data.dataset.age_gender_loader import create_loader
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging
from tqdm import tqdm


# Adjusted Mivolo loss, without weighting
class CombindMivoloLossAdjusted(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gnll = torch.nn.GaussianNLLLoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, age_pred, age_actual, age_var, gender_pred, gender_actual):
        return self.gnll(age_pred, age_actual, age_var) + self.cross_entropy(
            gender_pred, gender_actual
        )


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Training Adaption")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00001,
        required=False,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        required=False,
        help="Weight decay for the optimizer",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        required=True,
        help="Name of the checkpoint to save",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        required=False,
        help="Number of epochs to train for",
    )

    return parser


def main():
    parser = get_parser()
    user_args = parser.parse_args()

    # Loading the pre-trained model
    args_dict = {
        "output": "output",
        "detector_weights": "models/yolov8x_person_face.pt",
        "checkpoint": "models/mivolo_imdb_adjusted.pth.tar",
        "with_persons": False,
        "disable_faces": False,
        "draw": False,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "half": False,
    }

    args = argparse.Namespace(**args_dict)

    setup_default_logging()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)

    predictor = Predictor(args, verbose=True)

    # Loading the datasets
    train_dataset = AgeGenderDataset(
        "mivolo/data/dataset/images",
        "mivolo/data/dataset/annotations",
        name="train",
        split="train",
        use_persons=False,
        model_with_persons=False,
        is_training=True,
        min_age=predictor.age_gender_model.meta.min_age,
        max_age=predictor.age_gender_model.meta.max_age,
    )

    train_loader = create_loader(
        train_dataset,
        (3, 224, 224),
        176,
        num_workers=8,
        crop_pct=None,
        crop_mode=None,
        pin_memory=True,
        img_dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        persistent_workers=True,
        worker_seeding="all",
        target_type=torch.float,
    )

    valid_dataset = AgeGenderDataset(
        "mivolo/data/dataset/images",
        "mivolo/data/dataset/annotations",
        name="valid",
        split="valid",
        use_persons=False,
        model_with_persons=False,
        is_training=False,
        min_age=predictor.age_gender_model.meta.min_age,
        max_age=predictor.age_gender_model.meta.max_age,
    )

    valid_loader = create_loader(
        valid_dataset,
        (3, 224, 224),
        176,
        num_workers=8,
        crop_pct=None,
        crop_mode=None,
        pin_memory=True,
        img_dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        persistent_workers=True,
        worker_seeding="all",
        target_type=torch.float,
    )

    # Define your loss function
    criterion = CombindMivoloLossAdjusted()

    # Define your optimizer
    optimizer = optim.Adam(
        predictor.age_gender_model.model.parameters(),
        lr=user_args.learning_rate,
        weight_decay=user_args.weight_decay,
    )

    # Training loop
    with open(f"{user_args.name}_loss", "a") as out_file:
        for epoch in range(user_args.epochs):
            # Training phase
            predictor.age_gender_model.model.train()
            running_loss = 0.0

            total_samples = 0
            # break
            for inputs, labels in tqdm(train_loader):
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                output = predictor.age_gender_model.model(inputs)

                age_output = output[:, 2]
                age_variance = output[:, 3].exp()
                gender_output = output[:, :2]

                loss = criterion(
                    age_output,
                    labels[:, 0],
                    age_variance,
                    gender_output,
                    labels[:, 1].type(torch.LongTensor).to(torch.device("cpu")),
                )

                if torch.isnan(loss):
                    print("The loss became nan. Stopping...")
                    exit()

                out_file.write(f"Train Loss: {loss}\n")

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Compute training statistics
                running_loss += loss.item() * inputs.size(0)
                total_samples += labels.size(0)

            train_loss = running_loss / total_samples

            # Validation phase
            predictor.age_gender_model.model.eval()
            val_loss = 0.0

            val_total_samples = 0
            print("Validating...")
            with torch.no_grad():
                for inputs, labels in tqdm(valid_loader):
                    # Forward pass
                    output = predictor.age_gender_model.model(inputs)

                    age_output = output[:, 2]
                    age_variance = output[:, 3].exp()
                    gender_output = output[:, :2]

                    loss = criterion(
                        age_output,
                        labels[:, 0],
                        age_variance,
                        gender_output,
                        labels[:, 1]
                        .type(torch.LongTensor)
                        .to(
                            torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        ),
                    )
                    out_file.write(f"Validation Loss: {loss}\n")

                    # Compute validation statistics
                    val_loss += loss.item() * inputs.size(0)
                    val_total_samples += labels.size(0)
            val_loss /= val_total_samples

            # Print the training and validation metrics for each epoch
            print(
                f"Epoch {epoch + 1}/{user_args.epochs}, "
                f"Training Loss: {train_loss:.4f} "
                f"Validation Loss: {val_loss:.4f}"
            )

            temp_state = torch.load("models/mivolo_imdb_adjusted.pth.tar")
            temp_state["state_dict"] = predictor.age_gender_model.model.state_dict()
            if epoch == 0:
                print(temp_state["state_dict"])
            torch.save(
                temp_state,
                f"models/{user_args.name}_LR{user_args.learning_rate}_WD_{user_args.weight_decay}_EPOCHS{user_args.epochs}_{epoch}_{round(train_loss,4)}_{round(val_loss, 4)}.pth.tar",
            )
        # Training loop ends


if __name__ == "__main__":
    main()
