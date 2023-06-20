from datasets import load_dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import UnidentifiedImageError
import numpy as np
import random
import torch
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        x, y = data["color"], data["normal"]

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.transform is not None:
            x = self.transform(x)

        random.seed(seed)  # apply this seed to target tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.transform is not None:
            y = self.transform(y)

        # x, y = np.array(data["color"]), np.array(data["normal"])
        # x = x.transpose(2, 0, 1)
        # y = y.transpose(2, 0, 1)
        # x = x / 255.0
        # y = y / 255.0

        return x, y


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, image_size, batch_size, num_workers=4):
        super().__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        dataset = self.prepare_data()

        train_data_idx, val_data_idx = train_test_split(
            list(range(len(dataset))), test_size=0.2, random_state=42
        )

        # Some images are corrupted (not possible to read, or only 1 channel, so skip them)
        train_data = []
        for idx in train_data_idx:
            try:
                data = dataset[idx]
                if (
                    len(data["color"].getbands()) != 3
                    or len(data["normal"].getbands()) != 3
                ):
                    raise UnidentifiedImageError
                train_data.append(dataset[idx])

            except UnidentifiedImageError:
                print(f"Image {idx} is corrupted. Skipping...")

        val_data = []
        for idx in val_data_idx:
            try:

                data = dataset[idx]
                if (
                    len(data["color"].getbands()) != 3
                    or len(data["normal"].getbands()) != 3
                ):
                    raise UnidentifiedImageError
                val_data.append(dataset[idx])

            except UnidentifiedImageError:
                print(f"Image {idx} is corrupted. Skipping...")

        self.train_dataset = ImageDataset(
            train_data,
            transform=self.make_transform(mode="train"),
        )
        self.val_dataset = ImageDataset(
            val_data,
            transform=self.make_transform(mode="val"),
        )

    def make_transform(self, mode="train"):
        if mode == "train":
            try:
                resized_shape = int(self.image_size * 1.2)
            except TypeError:
                resized_shape = tuple([int(x * 1.2) for x in self.image_size])

            return transforms.Compose(
                [
                    transforms.Resize(resized_shape),
                    transforms.RandomCrop(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            )
        elif mode == "val":
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def prepare_data(self):
        return load_dataset("dream-textures/textures-color-normal-1k")["train"]
