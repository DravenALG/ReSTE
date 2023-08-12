import os
import time
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load(type='both', dataset='cifar10', data_path='/home/xiaoming/dataset', batch_size=256, batch_size_test=256,
         num_workers=0, device_id=0, use_dali=False):
    # dataset basic config
    param = {
        'cifar10': {'name': datasets.CIFAR10, 'size': 32,
                    'normalize': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]},
        'imagenet': {'name': datasets.ImageFolder, 'size': 224, 'val_size': 256,
                     'normalize': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]}}

    # choose a specific dataset
    data = param[dataset]

    # load dataset and add to dataloaders
    if dataset == "cifar10":
        # train transform
        train_transform = transforms.Compose([
            transforms.RandomCrop(data['size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*data['normalize']),
        ])
        # test transform
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data['normalize']),
        ])
        # train data
        trainset = data['name'](root=data_path,
                                train=True,
                                download=True,
                                transform=train_transform)
        # train loader
        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)
        # test data
        testset = data['name'](root=data_path,
                               train=False,
                               download=True,
                               transform=test_transform)
        # test loader
        test_loader = DataLoader(
            testset,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True)

        # return data loader
        if type == 'both':
            return train_loader, test_loader
        elif type == 'train':
            return train_loader
        elif type == 'val':
            return test_loader

    elif dataset == "imagenet":
        # train transform
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(data['size'], scale=(0.08, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*data['normalize']),
        ])

        # val transform
        val_transform = transforms.Compose([
            transforms.Resize(data['val_size']),
            transforms.CenterCrop(data['size']),
            transforms.ToTensor(),
            transforms.Normalize(*data['normalize']),
        ])

        # train data
        trainset = data['name'](root=os.path.join(data_path, "train"),
                                transform=train_transform)

        # train loader
        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True)

        # val data
        val_set = data['name'](root=os.path.join(data_path, "val"),
                               transform=val_transform)

        # val loader
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True)

        # return data loader
        if type == 'both':
            return train_loader, val_loader
        elif type == 'train':
            return train_loader
        elif type == 'val':
            return val_loader


if __name__ == '__main__':
    torchvision.datasets.CIFAR10("./data/cifar10", train=True, download=True)
