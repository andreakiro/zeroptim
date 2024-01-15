from typing import Callable
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from zeroptim.supported import __supported_datasets__


class VisionDataset:
    @staticmethod
    def make_loader(
        dataset_name: str,  # torchvision dataset name
        batch_size: int = 64,
        shuffle: bool = True,
        split: str = "train",
        root: str = "./data",  # root directory to save dataset
        use_transform: bool = True,
        **kwargs  # more specific parameters
    ) -> DataLoader:
        transform = VisionDataset.get_transform(
            dataset_name, split, use_transform=use_transform
        )

        if dataset_name == "svhn":
            # different handling for SVHN as it has a slightly different API
            dataset = __supported_datasets__[dataset_name](
                root=root, split=split, download=True, transform=transform
            )

        else:
            train_split = split == "train"
            dataset = __supported_datasets__[dataset_name](
                root=root, train=train_split, download=True, transform=transform
            )

        if dataset_name == "mnist-digits" and "digits" in kwargs:
            # filters the dataset to include only specified digits
            indices = [
                i for i, (_, label) in enumerate(dataset) if label in kwargs["digits"]
            ]
            dataset = Subset(dataset, indices)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def get_transform(
        dataset_name: str, split: str = "train", use_transform: bool = True
    ) -> Callable:
        if not use_transform:
            return transforms.Compose([transforms.ToTensor()])

        if dataset_name in ["cifar10", "cifar100"]:
            # Random horizontal flipping for data augmentation and cropping.
            # Normalization with mean and standard deviation.
            if split == "train":
                return transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
            else:
                return transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )

        if dataset_name in ["mnist-digits", "mnist-fashion"]:
            # Normalization is often sufficient.
            # Additional data augmentation might be used depending on the task.
            return transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )

        if dataset_name == "svhn":
            # Random horizontal flipping for data augmentation and cropping.
            # Normalization with mean and standard deviation.
            return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        # in case we miss a case
        return transforms.Compose([transforms.ToTensor()])
