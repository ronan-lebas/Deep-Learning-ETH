from torchvision import transforms
from dataset import CroppedImageDataset
import wandb
from train import train_model


wandb.init(
project="focusnet-bbox-refinement",
config={
    "model": "FocusNetv5",
    "batch_size": 8,
    "learning_rate": 0.065,
    "early_stop_patience": 15,
    "epochs": 150,
    "optimizer": "SGD",
    "loss_function": "L1+IOU",
    "train_split": 0.8,
    "mode": "basic_padding",
    "eval_mode": "basic_padding",
})

config = wandb.config

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CroppedImageDataset(mode=config.mode, split="train", transform=transform)
test_dataset = CroppedImageDataset(mode=config.eval_mode, split="test", transform=transform)

train_model(train_dataset, test_dataset, config)