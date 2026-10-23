import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import MNIST


def remove_zero_padding(arr):
    rows = [i for i in range(arr.shape[0]) if np.any(arr[i] != 0)]
    left, right = (rows[0], rows[-1]) if rows else (0, 0)
    return arr[:, left:right]


def concat(a, b, overlap, intersection_scalar):
    if not overlap:
        return np.concatenate((a, b), axis=1)
    intersection = int(intersection_scalar * min(a.shape[1], b.shape[1]))
    image = np.zeros((a.shape[0], a.shape[1] + b.shape[1] - intersection))
    image[:, : a.shape[1]][a != 0] = a[a != 0]
    shifted = image[:, a.shape[1] - intersection : a.shape[1] - intersection + b.shape[1]]
    shifted[b != 0] = b[b != 0]
    return image


def merge(digits):
    image = np.zeros((28, 1))
    for digit in digits:
        digit = remove_zero_padding(digit)
        intersection = random.randint(1, 30) / 100
        overlap = random.choice([True, False])
        image = concat(image, digit, overlap, intersection)
    return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("datasets/multimnist"))
    p.add_argument("--mnist-root", type=Path, default=Path("datasets/mnist"))
    p.add_argument("--train-samples", type=int, default=10000)
    p.add_argument("--test-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    random.seed(args.seed)
    counter = 0
    for split, n in [("train", args.train_samples), ("test", args.test_samples)]:
        folder = args.output / split / "2"
        folder.mkdir(parents=True, exist_ok=True)
        mnist = MNIST(str(args.mnist_root), train=split == "train", download=True)
        images, labels = mnist.data.numpy(), mnist.targets.numpy()
        indices = list(range(len(images)))
        with (folder.parent / "labels.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            for _ in range(n):
                chosen = random.choices(indices, k=2)
                image = np.concatenate([np.zeros((28, 2)), merge([images[i] for i in chosen]), np.zeros((28, 2))], axis=1)
                Image.fromarray(image.astype(np.uint8)).save(folder / f"{counter}.png")
                writer.writerow([f"{counter}.png", "".join(str(labels[i]) for i in chosen)])
                counter += 1


if __name__ == "__main__":
    main()
