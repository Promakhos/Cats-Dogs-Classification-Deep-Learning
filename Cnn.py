import torch
import torch.optim as optim
from data_utils import get_dataloaders
from model_Cnn import Cnn
from train_utils import train_model
from train_utils import plot_training_results
from torch.optim import lr_scheduler



if __name__ == '__main__':

    H = {"val_accu": [], "val_loss": []}

    num_epochs = 15
    # Constants
    data_dir = 'data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Get Dataloaders
    dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir)
    print(class_names)

    # Model Definition
    model = Cnn().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training
    train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs, dataloaders=dataloaders, device=device, dataset_sizes=dataset_sizes, H=H)

    # Visualization
    plot_training_results(H)

    
    
