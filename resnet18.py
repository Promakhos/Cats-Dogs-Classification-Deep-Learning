import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from data_utils import get_dataloaders
from train_utils import train_model
from train_utils import plot_training_results
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import models

if __name__ == '__main__':
    H = {"val_accu": [], "val_loss": []}

    num_epochs = 15
    data_dir = 'data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get Dataloaders
    dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir)

    # Model Definition
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training
    train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs, dataloaders=dataloaders, device=device, dataset_sizes=dataset_sizes, H=H)

    # Visualization
    plot_training_results(H)
