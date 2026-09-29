import os
import glob
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp"
)


class SRDataset(Dataset):

    def __init__(
        self,
        image_dir,
        patch_size=48,
        scale=4,
        augment=True,
        samples_per_image=20
    ):

        self.paths = sorted(
            p for p in glob.glob(
                os.path.join(image_dir, "**", "*"),
                recursive=True
            )
            if p.lower().endswith(EXTENSIONS)
        )

        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images found in {image_dir}"
            )

        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        self.samples_per_image = samples_per_image

        self.hr_size = patch_size * scale

    def __len__(self):
        return len(self.paths) * self.samples_per_image

    def __getitem__(self, index):

        image_path = self.paths[index % len(self.paths)]

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        if width < self.hr_size or height < self.hr_size:

            image = image.resize(
                (
                    max(width, self.hr_size),
                    max(height, self.hr_size)
                ),
                Image.Resampling.BICUBIC
            )

            width, height = image.size

        left = random.randint(
            0,
            width - self.hr_size
        )

        top = random.randint(
            0,
            height - self.hr_size
        )

        hr = image.crop(
            (
                left,
                top,
                left + self.hr_size,
                top + self.hr_size
            )
        )

        if self.augment:

            if random.random() < 0.5:
                hr = TF.hflip(hr)

            if random.random() < 0.5:
                hr = TF.vflip(hr)

            rotations = random.randint(0, 3)

            if rotations:
                hr = TF.rotate(
                    hr,
                    rotations * 90
                )

        lr = hr.resize(
            (
                self.patch_size,
                self.patch_size
            ),
            Image.Resampling.BICUBIC
        )

        lr = TF.to_tensor(lr)
        hr = TF.to_tensor(hr)

        return lr, hr


def make_train_val_split(
    image_dir,
    val_fraction=0.1,
    seed=42
):

    paths = sorted(
        p for p in glob.glob(
            os.path.join(image_dir, "**", "*"),
            recursive=True
        )
        if p.lower().endswith(EXTENSIONS)
    )

    if len(paths) < 2:
        raise ValueError(
            "At least 2 images are required."
        )

    random.Random(seed).shuffle(paths)

    n_val = max(
        1,
        int(len(paths) * val_fraction)
    )

    n_val = min(
        n_val,
        len(paths) - 1
    )

    return paths[n_val:], paths[:n_val]