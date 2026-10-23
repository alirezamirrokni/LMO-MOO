import csv
import fnmatch
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _crop_resize(x, i, j, h, w, size, mode):
    kwargs = {"align_corners": True} if mode == "bilinear" else {}
    return F.interpolate(x[None, :, i : i + h, j : j + w], size=size, mode=mode, **kwargs).squeeze(0)


class DenseDataset(Dataset):
    def __init__(self, root, train=True, augmentation=False, normals=False):
        self.root = os.path.join(root, "train" if train else "val")
        self.augmentation = augmentation
        self.normals = normals
        self.label_dir = "label" if normals else "label_7"
        self.length = len(fnmatch.filter(os.listdir(os.path.join(self.root, "image")), "*.npy"))

    def _load(self, folder, index, channels_last=True):
        array = np.load(os.path.join(self.root, folder, f"{index}.npy"))
        return torch.from_numpy(np.moveaxis(array, -1, 0) if channels_last else array)

    def __getitem__(self, index):
        image = self._load("image", index)
        semantic = self._load(self.label_dir, index, channels_last=False)
        depth = self._load("depth", index)
        normal = self._load("normal", index) if self.normals else None

        if self.augmentation:
            height, width = image.shape[-2:]
            scale = [1.0, 1.2, 1.5][random.randint(0, 2)]
            h, w = int(height / scale), int(width / scale)
            i, j = random.randint(0, height - h), random.randint(0, width - w)
            size = (height, width)
            image = _crop_resize(image, i, j, h, w, size, "bilinear")
            semantic = _crop_resize(semantic[None], i, j, h, w, size, "nearest").squeeze(0)
            depth = _crop_resize(depth, i, j, h, w, size, "nearest") / scale
            if self.normals:
                normal = _crop_resize(normal, i, j, h, w, size, "bilinear")
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                semantic = torch.flip(semantic, dims=[1])
                depth = torch.flip(depth, dims=[2])
                if self.normals:
                    normal = torch.flip(normal, dims=[2])
                    normal[0, :, :] = -normal[0, :, :]

        items = [image.float(), semantic.float(), depth.float()]
        if self.normals:
            items.append(normal.float())
        return tuple(items)

    def __len__(self):
        return self.length


class MultiMNIST(Dataset):
    def __init__(self, root, split="train"):
        transform = transforms.Compose([transforms.Resize((28, 43)), transforms.ToTensor()])
        folder = os.path.join(root, split)
        images, left, right = [], [], []
        with open(os.path.join(folder, "labels.csv")) as f:
            for row in csv.reader(f):
                if row:
                    images.append(transform(Image.open(os.path.join(folder, "2", row[0].strip())).convert("L")))
                    left.append(int(row[1].strip()[0]))
                    right.append(int(row[1].strip()[1]))
        self.images = torch.stack(images)
        self.left = torch.tensor(left, dtype=torch.long)
        self.right = torch.tensor(right, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.left[index], self.right[index]
