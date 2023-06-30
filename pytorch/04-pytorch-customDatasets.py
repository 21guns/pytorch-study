#!/usr/bin/env python
import torch
from torch import nn
from helper_functions import *
device = "cuda" if torch.cuda.is_available() else "cpu"

# import requests
# import zipfile
from pathlib import Path

data_path = Path("data")
image_path = data_path / "pizza_steak_sushi"

# if image_path.is_dir():
#     print(f"[INFO] {image_path} directory exists, skipping download.")
# else:
#     print(f"[INFO] Did not find {image_path} directory, creating one...")
#     image_path.mkdir(parents=True, exist_ok=True)
#     with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
#         f.write(requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip").content)
#     with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
#         zip_ref.extractall(data_path)

train_dir = image_path / "train"
test_dir = image_path / "test"

import random
from PIL import Image

random.seed(42)
image_path_list = list(image_path.glob("*/*/*.jpg"))
randm_image_path = random.choice(image_path_list)
image_class = randm_image_path.parent.stem
img = Image.open(randm_image_path)
print(f"Image path: {randm_image_path}")
print(f"Image class: {image_class}")
# img.show()

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from typing import Tuple, Dict, List

data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# plot_transformed_images(image_path_list, data_transform, 3)
# plt.show()
train_data = datasets.ImageFolder(train_dir, transform=data_transform, target_transform=None)
test_data = datasets.ImageFolder(test_dir, transform=data_transform)
print(f"{train_data}\n {test_data}")

from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_data, batch_size=1, num_workers=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, num_workers=1, shuffle=False)


class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None) -> None:
        super().__init__()
        self.paths = list(Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.calss_to_idx = find_classes(targ_dir)
    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)
        
    def __len__(self) -> int:
        return len(self.paths)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img = self.load_image(idx)
        calss_name =  self.paths[idx].parent.name
        class_idx = self.calss_to_idx[calss_name]
        if self.transform:
            return self.transform(img), class_idx
        return img, calss_name

train_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])
train_data_custom = ImageFolderCustom(train_dir, transform=train_transform)
test_data_custom = ImageFolderCustom(test_dir, transform=test_transform)

# display_random_images(train_data, n=5, classes=train_data.classes, seed=None)
# display_random_images(train_data_custom, n=12, classes=train_data.classes, seed=None)

trian_dataloader_custom = DataLoader(train_data_custom, batch_size=1, num_workers=0, shuffle=True)
test_dataloader_custom = DataLoader(test_data_custom, batch_size=1, num_workers=0, shuffle=False)

train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

image_path_list = list(image_path.glob("*/*/*.jpg"))
# plot_transformed_images(image_path_list, train_transform, n=3, seed=None)

simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])
