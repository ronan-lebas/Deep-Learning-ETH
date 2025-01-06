import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CroppedImageDataset, data_transform
from FocusNet import FocusNet
from utils import display_resized_image_with_bbox
import wandb  # Weights & Biases for tracking
from tqdm import tqdm

# Initialize W&B project
wandb.init(
    project="focusnet-bbox-refinement",
    config={
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 100,
        "optimizer": "Adam",
        "loss_function": "SmoothL1Loss",
        "train_split": 0.8
    }
)
config = wandb.config

def train_model():
    # Paths
    train_image_folder = "dataset_cropped/train/"
    train_annotations_file = "dataset_cropped/instances_train.json"
    test_image_folder = "dataset_cropped/val/"
    test_annotations_file = "dataset_cropped/instances_val.json"

    # Load full dataset
    full_train_dataset = CroppedImageDataset(train_image_folder, train_annotations_file, transform=data_transform)

    # Split dataset into train (80%) and validation (20%)
    train_size = int(config.train_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Load test dataset
    test_dataset = CroppedImageDataset(test_image_folder, test_annotations_file, transform=data_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = FocusNet()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def evaluate(loader):
        """Evaluate the model on a given DataLoader."""
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, bboxes in loader:
                images = images.to(device)
                bboxes = bboxes.to(device)
                outputs = model(images)
                loss = criterion(outputs, bboxes)
                total_loss += loss.item()
        return total_loss / len(loader)

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{config.epochs}", unit="batch") as pbar:
            for batch_idx, (images, bboxes) in enumerate(train_loader):
                # Move data to device
                images = images.to(device)
                bboxes = bboxes.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, bboxes)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate training loss
                train_loss += loss.item()

                # Update progress bar
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        # Average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)

        # Evaluate on validation set
        val_loss = evaluate(val_loader)

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss
        })

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model locally
    model_path = "focusnet_refined_bbox.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved at {model_path}")

    # Final evaluation on test dataset
    test_loss = evaluate(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}")
    wandb.log({"test_loss": test_loss})

if __name__ == "__main__":
    train_model()