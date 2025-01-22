import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from FocusNet import FocusNet
from FocusNetv2 import FocusNetv2
from FocusNetv3 import FocusNetv3
from FocusNetv4 import FocusNetv4
import wandb
from tqdm import tqdm
import os
import torch.nn.functional as F


def custom_loss(pred, target, lmbda=2e-3):
    # Smooth L1 loss
    l1_loss = F.smooth_l1_loss(pred, target)

    # IoU Loss
    pred_box = pred.clone()
    target_box = target.clone()
    intersection = torch.min(pred_box[:, :2], target_box[:, :2]) * torch.min(
        pred_box[:, 2:], target_box[:, 2:]
    )
    union = torch.max(pred_box[:, :2], target_box[:, :2]) * torch.max(
        pred_box[:, 2:], target_box[:, 2:]
    )
    iou = (intersection / (union + 1e-6)).mean()
    iou_loss = 1 - iou

    return l1_loss + lmbda * iou_loss

def train_model(full_train_dataset, test_dataset, config):
    # Split dataset into train (80%) and validation (20%)
    train_size = int(config.train_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    if config.model == "FocusNet":
        model = FocusNet()
    elif config.model == "FocusNetv2":
        model = FocusNetv2()
    elif config.model == "FocusNetv3":
        model = FocusNetv3()
    elif config.model == "FocusNetv4":
        model = FocusNetv4()
    else:
        raise ValueError("Unsupported model. Use 'FocusNet', 'FocusNetv2', 'FocusNetv3', or 'FocusNetv4'.")
    
    #criterion = nn.SmoothL1Loss()
    criterion = custom_loss

    if config.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    elif config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError("Unsupported optimizer. Use 'SGD' or 'Adam'.")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def evaluate(loader):
        """Evaluate the model on a given DataLoader."""
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, bboxes, image_ids in loader:
                images = images.to(device)
                bboxes = bboxes.to(device)
                outputs = model(images)
                loss = criterion(outputs, bboxes)
                total_loss += loss.item()
        return total_loss / len(loader)

    best_val_loss = float("inf")
    early_stop_counter = 0

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            unit="batch",
        ) as pbar:
            for batch_idx, (images, bboxes, image_ids) in enumerate(train_loader):
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

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        # Log metrics to W&B
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model
            model_path = f"{config.model}_M{config.mode}_L{config.loss_function}_O{config.optimizer}_LR{config.learning_rate}_E{config.epochs}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved as {model_path}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= config.early_stop_patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")

    # Final evaluation on test dataset
    model.load_state_dict(torch.load(model_path))  # Load the best model
    test_loss = evaluate(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}")
    wandb.log({"test_loss": test_loss})
