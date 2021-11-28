from pathlib import Path
from typing import Tuple, Optional

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def cifar10(folder: Optional[Path] = None) -> Tuple[Dataset, Dataset]:
    if folder is None:
        folder = Path('./data')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(folder, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(folder, train=False, download=True, transform=transform_test)
    return train_set, test_set


def cifar100(folder: Optional[Path] = None) -> Tuple[Dataset, Dataset]:
    if folder is None:
        folder = Path('./data')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
    ])

    train_set = torchvision.datasets.CIFAR100(folder, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(folder, train=False, download=True, transform=transform_test)
    return train_set, test_set


def imagenet(folder: Optional[Path] = None) -> Tuple[Dataset, Dataset]:
    if folder is None:
        folder = Path.home().joinpath('imagenet')
    train_dir = folder.joinpath('train')
    test_dir = folder.joinpath('val')

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4811, 0.4575, 0.4078), (0.2335, 0.2294, 0.2302)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4811, 0.4575, 0.4078), (0.2335, 0.2294, 0.2302)),
    ])

    train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    test_set = torchvision.datasets.ImageFolder(test_dir, transform=transform_test)
    return train_set, test_set


def fake_imagenet(folder: Optional[Path] = None) -> Tuple[Dataset, Dataset]:
    train_set = torchvision.datasets.FakeData(size=1281167, transform=transforms.ToTensor(), num_classes=1000)
    test_set = torchvision.datasets.FakeData(size=50000, transform=transforms.ToTensor(), num_classes=1000)
    return train_set, test_set
