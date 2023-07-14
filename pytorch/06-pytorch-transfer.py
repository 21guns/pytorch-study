from pathlib import Path
import data_setup
from torchvision import transforms, datasets
import torch
from helper_functions import *


data_path = Path("data")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

manual_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataloader,test_dataloader,class_names =data_setup.create_dataloaders(
    train_dir,
    test_dir,
    manual_transform,
    batch_size=32
)

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)

for param in model.features.parameters():
    param.requires_grad = False

output_shape = len(class_names)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280, out_features=output_shape, bias=True)
)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
torch.manual_seed(42)

from timeit import default_timer as timer
start_time = timer()
results = train(model, train_dataloader, test_dataloader, loss_fn, optimizer)
end_time = timer()
print(f"Runtime: {end_time - start_time} seconds")
# plot_loss_curves(results)
# plt.show()

# import random
num_images_to_plot = 3
test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))
test_image_path_sample = random.sample(test_image_path_list, num_images_to_plot)

for image_path in test_image_path_sample:
    pred_and_plot_image_v2(model, image_path, class_names, image_size=(224, 224))

custom_image_path = data_path / "04-pizza-dad.jpeg"
pred_and_plot_image_v2(model, custom_image_path, class_names, image_size=(224, 224))
plt.show()

