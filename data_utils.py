import torch
from torchvision import datasets, transforms
import os

def get_dataloaders(data_dir, batch_size=4, num_workers=2):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'training_set': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test_set': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['training_set', 'test_set']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
                   for x in ['training_set', 'test_set']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['training_set', 'test_set']}
    class_names = image_datasets['training_set'].classes
    

    return dataloaders, dataset_sizes, class_names
