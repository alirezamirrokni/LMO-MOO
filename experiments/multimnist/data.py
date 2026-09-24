# experiments/multimnist/data.py
import os
import csv
from typing import List, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MultiMNISTDataset(Dataset):
    """
    root/
      train/
        2/
          *.png
        labels.csv
      test/
        2/
          *.png
        labels.csv

    labels.csv: "0.png,21" -> 左=2, 右=1
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        assert split in ("train", "test")
        self.root = root
        self.split = split
        
        if transform is None:
            # 统一所有图片到 28x43，再转成 [1, 28, 43] 的 tensor
            self.transform = transforms.Compose(
                [
                    transforms.Resize((28, 43)),  # (height, width)
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

        split_dir = os.path.join(root, split)
        csv_path = os.path.join(split_dir, "labels.csv")
        img_dir = os.path.join(split_dir, "2")  # 目前就用 2 这个子目录

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"labels.csv not found: {csv_path}")

        self.img_dir = img_dir
        self.samples: List[Tuple[str, int, int]] = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                img_name = row[0].strip()
                label_str = row[1].strip()
                # 假设就是两个数字，例如 "21"
                if len(label_str) != 2:
                    raise ValueError(f"Label '{label_str}' is not 2 digits.")
                left_digit = int(label_str[0])
                right_digit = int(label_str[1])
                self.samples.append((img_name, left_digit, right_digit))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {csv_path}")

        print(
            f"[MultiMNISTDataset] split={split}, num_samples={len(self.samples)}, "
            f"img_dir={self.img_dir}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_name, left_digit, right_digit = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img = Image.open(img_path).convert("L")  # 灰度
        if self.transform is not None:
            img = self.transform(img)  # -> [1, 28, 43] float32

        # label 保持为 python int，DataLoader 默认会拼成 LongTensor
        return img, left_digit, right_digit


def get_dataloaders(
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 4,
):
    train_dataset = MultiMNISTDataset(root=data_root, split="train")
    test_dataset = MultiMNISTDataset(root=data_root, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


class CachedDataset(Dataset):
    """Materialize the existing deterministic transform without RNG changes.

    Used only with MultiMNISTDataset's default Resize + ToTensor pipeline.
    The default 10k dataset uses about 46 MiB of image storage in RAM.
    """
    def __init__(self, dataset):
        print(f'Caching {len(dataset)} preprocessed images in host RAM...', flush=True)
        rows = [dataset[i] for i in range(len(dataset))]
        self.images = torch.stack([row[0] for row in rows])
        self.left = torch.tensor([row[1] for row in rows], dtype=torch.long)
        self.right = torch.tensor([row[2] for row in rows], dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.left[index], self.right[index]
