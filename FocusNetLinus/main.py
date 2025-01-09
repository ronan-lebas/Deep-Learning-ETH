from training import train
from eval2 import eval
import torch
import os
from focusnet import FocusNetROI

os.chdir('dataset/instance_version')



expansion_ratio = 0.5
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FocusNetROI().to(device)

    # train(expansion_ratio, device)
    eval(0.5, device,model)
