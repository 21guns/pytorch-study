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
# print(f"{train_data}\n {test_data}")

# from torch.utils.data import DataLoader
# train_dataloader = DataLoader(train_data, batch_size=1, num_workers=1, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=1, num_workers=1, shuffle=False)


# class ImageFolderCustom(Dataset):
#     def __init__(self, targ_dir: str, transform=None) -> None:
#         super().__init__()
#         self.paths = list(Path(targ_dir).glob("*/*.jpg"))
#         self.transform = transform
#         self.classes, self.calss_to_idx = find_classes(targ_dir)
#     def load_image(self, index: int) -> Image.Image:
#         image_path = self.paths[index]
#         return Image.open(image_path)
        
#     def __len__(self) -> int:
#         return len(self.paths)
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
#         img = self.load_image(idx)
#         calss_name =  self.paths[idx].parent.name
#         class_idx = self.calss_to_idx[calss_name]
#         if self.transform:
#             return self.transform(img), class_idx
#         return img, calss_name

# train_transform = transforms.Compose([
#     transforms.Resize(size=(64, 64)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor()
# ])
# test_transform = transforms.Compose([
#     transforms.Resize(size=(64, 64)),
#     transforms.ToTensor()
# ])
# train_data_custom = ImageFolderCustom(train_dir, transform=train_transform)
# test_data_custom = ImageFolderCustom(test_dir, transform=test_transform)

# # display_random_images(train_data, n=5, classes=train_data.classes, seed=None)
# # display_random_images(train_data_custom, n=12, classes=train_data.classes, seed=None)

# trian_dataloader_custom = DataLoader(train_data_custom, batch_size=1, num_workers=0, shuffle=True)
# test_dataloader_custom = DataLoader(test_data_custom, batch_size=1, num_workers=0, shuffle=False)

# train_transform = transforms.Compose([
#     transforms.Resize(size=(224, 224)),
#     transforms.TrivialAugmentWide(num_magnitude_bins=31),
#     transforms.ToTensor()
# ])
# test_transform = transforms.Compose([
#     transforms.Resize(size=(224, 224)),
#     transforms.ToTensor()
# ])

# image_path_list = list(image_path.glob("*/*/*.jpg"))
# plot_transformed_images(image_path_list, train_transform, n=3, seed=None)

simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])
train_data_simple = datasets.ImageFolder(train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(test_dir, transform=simple_transform)
BATCH_SIZE = 1
NUM_WORKERS = 0 #os.cpu_count()
train_dataloader_simple = DataLoader(train_data_simple, batch_size=BATCH_SIZE, shuffle=True,
                                     num_workers=NUM_WORKERS)
test_dataloader_simple = DataLoader(test_data_simple, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS)

class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*16*16, output_shape)
        )
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
    
torch.manual_seed(42)
NUM_EPOCHS = 5

# model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data.classes))
        
# from torchinfo import summary
# summary(model_0, (1, 3, 64, 64))

# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)

from timeit import default_timer as timer
# start_time = timer()

# model_0_results = train(model_0, train_dataloader_simple, test_dataloader_simple, loss_fn, optimizer, NUM_EPOCHS)

# end_time = timer()
# print(f"Model 0 took {end_time - start_time} seconds to train")
# plot_loss_curves(model_0_results)
# plt.show()

#model_1
train_transform_tivial_augment = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

train_data_augment = datasets.ImageFolder(train_dir, transform=train_transform_tivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)

train_dataloader_augment = DataLoader(train_data_augment, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader_simple = DataLoader(test_data_simple, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model_1 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data_augment.classes))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.001)

start_time = timer()
model_1_results = train(model_1, train_dataloader_augment, test_dataloader_simple, loss_fn, optimizer, NUM_EPOCHS)
end_time = timer()
print(f"Model 1 took {end_time - start_time} seconds to train")
# plot_loss_curves(model_1_results)
# plt.show()

custom_image_path = image_path / "04-pizza-dad.jpeg"
import torchvision
custom_image_uint8 = torchvision.io.read_image(custom_image_path)
