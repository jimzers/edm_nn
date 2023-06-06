"""
Dataset loading functions.
"""

from typing import Text, Optional

import torch
import torchvision


def load_dataset(
        dataset_name: Text = 'mnist',
        data_dir: Optional[Text] = './data',
        batch_size: Optional[int] = 256,
        shuffle_train: Optional[bool] = True,
        shuffle_test: Optional[bool] = False,
        num_workers: Optional[int] = 1,
        flatten: Optional[bool] = False,
):
    """
    Loads the dataset.

    Args:
        dataset_name: name of the dataset to load
        data_dir: directory to save the dataset to

    Returns:
        train_loader: training data loader
        test_loader: test data loader

    """

    # send to tensor, then flatten
    transform_arr = [
        torchvision.transforms.ToTensor(),
    ]

    if flatten:
        transform_arr += [torchvision.transforms.Lambda(lambda x: x.view(-1))]

    transform_fn = torchvision.transforms.Compose(transform_arr)

    if dataset_name == 'mnist':

        # Load MNIST dataset
        train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True,
                                                   transform=transform_fn,
                                                   download=True)
        test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                                  transform=transform_fn,
                                                  download=True)


    elif dataset_name == 'fashion_mnist':

        # Load Fashion MNIST dataset
        train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True,
                                                          transform=transform_fn,
                                                          download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False,
                                                         transform=transform_fn,
                                                         download=True)


    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers
    )

    return train_loader, test_loader
