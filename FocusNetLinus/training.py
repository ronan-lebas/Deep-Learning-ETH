import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from focusnet import FocusNetDataset, FocusNetROI


def train(expansion_ratio, device, model):
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset_train = FocusNetDataset(
        annotation_file='instances_train_trashcan_expanded.json',
        image_dir='train',
        transform=transform,
        expansion_ratio=expansion_ratio
    )

    dataloader = DataLoader(dataset_train, batch_size=16, shuffle=True)

    dataset_val = FocusNetDataset(
        annotation_file='instances_val_trashcan_expanded.json',
        image_dir='val',
        transform=transform,
        expansion_ratio=expansion_ratio
    )

    val_loader = DataLoader(dataset_val, batch_size=16, shuffle=True)

    # Model, Loss, Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Configuration
    num_epochs = 70
    patience = 8  # Number of epochs to wait for improvement
    min_delta = 1e-4  # Minimum improvement to reset patience counter
    save_path = 'best_focusnet_model.pth'

    # Track Training and Validation Metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    best_model_wts = None  # Placeholder for the best model weights
    best_epoch = 0
    epochs_without_improvement = 0  # Early stopping counter

    # Custom IoU Loss Example
    def iou_loss(predicted, target):
        x1_p, y1_p, w_p, h_p = predicted[:, 0], predicted[:, 1], predicted[:, 2], predicted[:, 3]
        x1_t, y1_t, w_t, h_t = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        x2_p = x1_p + w_p
        y2_p = y1_p + h_p
        x2_t = x1_t + w_t
        y2_t = y1_t + h_t

        inter_x1 = torch.max(x1_p, x1_t)
        inter_y1 = torch.max(y1_p, y1_t)
        inter_x2 = torch.min(x2_p, x2_t)
        inter_y2 = torch.min(y2_p, y2_t)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        area_p = w_p * h_p
        area_t = w_t * h_t

        union_area = area_p + area_t - inter_area
        iou = inter_area / (union_area + 1e-6)

        return 1 - iou.mean()

    criterion_l1 = nn.SmoothL1Loss()

    def combined_loss(predicted, target):
        """
        Combined Smooth L1 Loss and IoU Loss.
        """
        l1_loss = criterion_l1(predicted, target)
        iou = iou_loss(predicted, target)
        return l1_loss + iou

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train_samples = 0

        # Training Loop
        for images, expanded_bboxes, original_bboxes in tqdm(dataloader,
                                                             desc=f"Epoch {epoch + 1}/{num_epochs} - Training (Expansion {expansion_ratio})"):
            images = images.to(device)
            expanded_bboxes = expanded_bboxes.to(device)
            original_bboxes = original_bboxes.to(device)

            optimizer.zero_grad()
            outputs = model(images, expanded_bboxes)
            loss = combined_loss(outputs, original_bboxes)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            total_train_samples += images.size(0)

        avg_train_loss = train_loss / total_train_samples

        # Validation Loop
        model.eval()
        val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for images, expanded_bboxes, original_bboxes in tqdm(val_loader,
                                                                 desc=f"Validation (Expansion {expansion_ratio})"):
                images = images.to(device)
                expanded_bboxes = expanded_bboxes.to(device)
                original_bboxes = original_bboxes.to(device)

                outputs = model(images, expanded_bboxes)
                loss = combined_loss(outputs, original_bboxes)

                val_loss += loss.item() * images.size(0)
                total_val_samples += images.size(0)

        avg_val_loss = val_loss / total_val_samples

        # Track Metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")

        # Check for Improvement
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_wts = model.state_dict()
            best_epoch = epoch + 1
            epochs_without_improvement = 0  # Reset early stopping counter
            print(f"Best model updated at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        # Early Stopping Check
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # Save the best model at the end of training
    if best_model_wts is not None:
        torch.save(best_model_wts, save_path)
        print(f"\nBest model saved to '{save_path}' with validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    else:
        print("No model was saved, as no improvement was observed.")
