#!/usr/bin/env python -i
from timeit import default_timer as timer
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from helper_functions import *
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
    target_transform=None
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)
class_names = train_data.classes

# imgae, label = train_data[0]
# print(imgae.shape)
# plt.imshow(imgae.squeeze())
# plt.title(label)
# plt.show()

BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
train_features_batch, train_labels_batch = next(iter(train_dataloader))

class FashionMNISTModelV0(torch.nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_shape, hidden_units),
            torch.nn.Linear(hidden_units, output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)

class FashionMNISTModelV1(torch.nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_shape, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)

class FashionMNISTModelV2(torch.nn.Module):
    """
    Model architecture copy TinyVGG from:https://poloclub.github.io/cnn-explainer/
    """   
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifer(x)
        return x
    
model_0 = FashionMNISTModelV0(
    input_shape=28*28,
    hidden_units=10,
    output_shape=len(class_names)
)
model_0.to('cpu')
model_1 = FashionMNISTModelV1(
    input_shape=28*28,
    hidden_units=10,
    output_shape=len(class_names)
).to(device)
model_2 = FashionMNISTModelV2(
    input_shape=1,
    hidden_units=10,
    output_shape=len(class_names)
).to(device)

torch.manual_seed(42)

epochs = 3

"""
model_0
"""
# train_time_start_on_cpu = timer()
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)
# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch}")
#     train_loss = 0
#     for batch, (X, y) in enumerate(train_dataloader):
#         model_0.train()
#         y_pred = model_0(X)
#         loss = loss_fn(y_pred, y)
#         train_loss += loss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # if batch % 400 == 0:
#         #     print(f"look at {batch * len(X)}/{len(train_dataloader.dataset)}")

#     train_loss /= len(train_dataloader)

#     test_loss, test_acc = 0, 0
#     model_0.eval()
#     with torch.inference_mode():
#         for X, y in test_dataloader:
#             test_pred = model_0(X)
#             test_loss += loss_fn(test_pred, y)
#             test_acc += accuracy_fn(y, test_pred.argmax(dim=1))

#         test_loss /= len(test_dataloader)
#         test_acc /= len(test_dataloader)
#     print(f"Train loss: {train_loss}, Test loss: {test_loss}, Test acc: {test_acc}")

# train_time_end_on_cpu = timer()
# total_train_time_model_0 = print_train_time(train_time_start_on_cpu, train_time_end_on_cpu)
# device = str(next(model_0.parameters()).device)
# model_0_results  =  eval_model(model_0, test_dataloader, loss_fn, accuracy_fn)
# print(model_0_results)

"""
model_1
# """
# train_time_start_on_cpu = timer()
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)
# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch}")
#     train_step(model_1, train_dataloader, loss_fn, optimizer, accuracy_fn)
#     test_step(model_1, test_dataloader, loss_fn, accuracy_fn)

# train_time_end_on_cpu = timer()
# total_train_time_model_1 = print_train_time(train_time_start_on_cpu, train_time_end_on_cpu)

# model_1_results  =  eval_model(model_1, test_dataloader, loss_fn, accuracy_fn)
# print(model_1_results)

"""
model_2
"""
train_time_start_on_cpu = timer()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}")
    train_step_accuary(model_2, train_dataloader, loss_fn, optimizer, accuracy_fn)
    test_step_accuary(model_2, test_dataloader, loss_fn, accuracy_fn)

train_time_end_on_cpu = timer()
total_train_time_model_2 = print_train_time(train_time_start_on_cpu, train_time_end_on_cpu)
model_2_results  =  eval_model(model_2, test_dataloader, loss_fn, accuracy_fn)
print(model_2_results)

# images = torch.randn(size=(32, 3, 64, 64))
# test_image = images[0]
# print(test_image.shape)

# conv_layer = torch.nn.Conv2d(in_channels=3, out_channels=10, 
#                              kernel_size=3, stride=1, padding=0)
# print(conv_layer(test_image).shape)

# 显示模型的训练得分
# import pandas as pd
# compare_results = pd.DataFrame([model_0_results, model_1_results, mode_2_results])
# compare_results["training_time"] = [total_train_time_model_0, total_train_time_model_1, total_train_time_model_2]

# compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
# plt.xlabel("Accuracy %")
# plt.ylabel("Model")
# plt.show()


# 使用model_2进行预测
# import random
# random.seed(42)
# test_samples = []
# test_labels = []
# for sample, label in random.sample(list(test_data), k=9):
#     test_samples.append(sample)
#     test_labels.append(label)

# pred_probs = make_predictions(model_2, test_samples)
# pred_classes = pred_probs.argmax(dim=1)
# print(pred_classes)
# plt.figure(figsize=(9, 9))
# nrows = 3
# ncols = 3
# for i, sample in enumerate(test_samples):
#     plt.subplot(nrows, ncols, i+1)
#     plt.imshow(sample.squeeze(), cmap="gray")
#     pred_label = class_names[pred_classes[i]]
#     truth_label = class_names[test_labels[i]]
#     title_text = f"pred: {pred_label}, truth: {truth_label}"
#     if pred_label == truth_label:
#         color = "green"
#     else:
#         color = "red"
#     plt.title(title_text, color=color)
#     plt.axis(False)
# plt.show()

# 使用混淆矩阵评估
# y_preds = []
# model_2.eval()
# with torch.inference_mode():
#     for X, y in tqdm(test_dataloader, desc="making predictions"):
#         X, y = X.to(device), y.to(device)
#         y_logit= model_2(X)
#         y_pred =  torch.softmax(y_logit, dim=1).argmax(dim=1)
#         y_preds.append(y_pred)
# y_pred_tensor = torch.cat(y_preds)

# from torchmetrics import ConfusionMatrix
# from mlxtend.plotting import plot_confusion_matrix

# confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
# confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets)

# fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
#                                 class_names=class_names,
#                                 figsize=(10, 7))
# plt.show()

# 保存模型
from pathlib import Path
MODEL_PATH =  Path("models")
MODEL_PATH.mkdir(parents=True,  exist_ok=True)
MODEL_NAME = "03-pytorch-vision-model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# torch.save(model_2.state_dict(), MODEL_SAVE_PATH)

loaded_model_2 = FashionMNISTModelV2(
    input_shape=1,
    hidden_units=10,
    output_shape=len(class_names)
)
loaded_model_2.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model_2_results  =  eval_model(loaded_model_2, test_dataloader, loss_fn, accuracy_fn)

close =  torch.isclose(torch.tensor(model_2_results["model_loss"]), 
              torch.tensor(loaded_model_2_results["model_loss"]),
              atol=1e-08,
              rtol=0.0001)
print(close)