from pathlib import Path
from torchvision import datasets, transforms
from helper_functions import *
import data_setup
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

data_path = Path("data")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
    ])
print(f"Manual transforms: {manual_transforms}")


weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
automatic_transforms = weights.transforms()
print(f"Automatic transforms: {automatic_transforms}")

train_dataloader, test_dataloader,  class_names = data_setup.create_dataloaders(
    train_dir, test_dir, automatic_transforms, batch_size=32)
model = torchvision.models.efficientnet_b0(weights=weights)

for param in model.features.parameters():
    param.requires_grad = False

set_seeds()
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(1280, len(class_names), bias=True)
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int = 5,
          device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
          writer: SummaryWriter = None) :
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Train Acc: {train_acc}, Test Loss: {test_loss}, Test Acc: {test_acc}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if  writer :
            writer.add_scalars("loss", {"train": train_loss, "test": test_loss}, epoch)
            writer.add_scalars("accuracy", {"train": train_acc, "test": test_acc}, epoch)
            writer.close()   
        else:
            pass 
    # writer.add_graph(model, torch.rand(1, 3, 224, 224))
    return results

set_seeds()
results = train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=5)

# Download 10 percent and 20 percent training data (if necessary)
data_10_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                                     destination="pizza_steak_sushi")

data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                     destination="pizza_steak_sushi_20_percent")

train_dir_10_percent = data_10_percent_path / "train"
train_dir_20_percent = data_20_percent_path / "train"
test_dir =  data_10_percent_path / "test"

BATCH_SIZE = 32

train_dataloader_10_percent, test_dataloader_10_percent, class_names = data_setup.create_dataloaders(
    train_dir_10_percent, test_dir, automatic_transforms, batch_size=BATCH_SIZE)
train_dataloader_20_percent, test_dataloader_20_percent, class_names = data_setup.create_dataloaders(
    train_dir_20_percent, test_dir, automatic_transforms, batch_size=BATCH_SIZE)

import torchvision
from torchinfo import summary
from torch import nn

effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
effnetb2 = torchvision.models.efficientnet_b2(weights=effnetb2_weights)

OUT_FEATURES = len(class_names)

def create_effnetb0():
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False
    
    set_seeds()
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(1280, OUT_FEATURES)
    )

    model.name = "EfficientNetB0"
    return model
def create_effnetb2():
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False
    
    set_seeds()
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(1408, OUT_FEATURES)
    )

    model.name = "EfficientNetB2"
    return model

num_epochs = [5, 10]
models = ["EfficientNetB0", "EfficientNetB2"]

train_dataloaders = {"10_percent": train_dataloader_10_percent, "20_percent": train_dataloader_20_percent}

experiment_number = 0

for dataloader_name, train_dataloader in train_dataloaders.items():
    for epoch in num_epochs:
        for model_name in models:
            experiment_number += 1
            print(f"Experiment number: {experiment_number}")
            print(f"Model: {model_name}")
            print(f"Epoch: {epoch}")
            print(f"Dataloader: {dataloader_name}")

            if model_name == "EfficientNetB0":
                model = create_effnetb0()
            elif model_name == "EfficientNetB2":
                model = create_effnetb2()

            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=epoch, 
                  writer=create_writer(dataloader_name, model_name, f"{epoch}_epochs"))
            
            # save_model()
            print("-" * 50 + "\n")