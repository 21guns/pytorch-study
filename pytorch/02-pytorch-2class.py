#!/usr/bin/env python
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from helper_functions import *

'''
    二元分类
'''
X, y = make_circles(n_samples=1000, noise=0.03, random_state=42)

# print(X.shape, y.shape)

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

# print(X[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

device = "cuda" if torch.cuda.is_available() else "cpu"

class CircleModelV0(torch.nn.Module):
    def __init__(self):
        super(CircleModelV0, self).__init__()
        self.linear_layer1 = torch.nn.Linear(2, 5)
        self.linear_layer2 = torch.nn.Linear(5, 1) 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer2(self.linear_layer1(x))

class CircleModelV1(torch.nn.Module):
    def __init__(self):
        super(CircleModelV1, self).__init__()
        self.linear_layer1 = torch.nn.Linear(2, 10)
        self.linear_layer2 = torch.nn.Linear(10, 10)
        self.linear_layer3 = torch.nn.Linear(10, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer3(self.linear_layer2(self.linear_layer1(x)))
    
class CircleModelV2(torch.nn.Module):
    def __init__(self):
        super(CircleModelV2, self).__init__()
        self.linear_layer1 = torch.nn.Linear(2, 10)
        self.linear_layer2 = torch.nn.Linear(10, 10)
        self.linear_layer3 = torch.nn.Linear(10, 1)
        self.relu  = torch.nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer3(self.relu(self.linear_layer2(self.relu(self.linear_layer1(x)))))

# model_0 = CircleModelV0().to(device)
model_0 = CircleModelV2().to(device)

# model_0 = torch.nn.Sequential(
#     torch.nn.Linear(2, 5),
#     torch.nn.Linear(5, 1)
# ).to(device)

untrained_preds = model_0(X_test.to(device))

loss_func = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_true)) * 100

# y_logits = model_0(X_test.to(device))[:5]
# y_pred_probs = torch.sigmoid(y_logits)

torch.manual_seed(42)

epochs = 1000
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_func(y_logits, y_train)
    acc =  accuracy_fn(y_train, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_func(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)
        # if epoch % 10 == 0:
        #     print(f'epoch {epoch} train loss: {loss.item():.3f} test loss: {test_loss.item():.3f} test acc: {test_acc:.3f}')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()