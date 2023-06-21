#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import *

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X, y = make_blobs(n_samples=1000, n_features=2, centers=4, cluster_std=1.5, random_state=42)

X = torch.tensor(X, dtype=torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

device = "cuda" if torch.cuda.is_available() else "cpu"

class BlobModel(torch.nn.Module):
    def __init__(self, input_features, output_features, hidden_units = 8):
        super(BlobModel, self).__init__()
        self.linear_layerr_stack = torch.nn.Sequential(
            torch.nn.Linear(input_features, hidden_units),
            torch.nn.Linear(hidden_units, hidden_units),
            torch.nn.Linear(hidden_units, output_features)
        )
    def forward(self, x):
        return self.linear_layerr_stack(x)
    
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_true)) * 100

model_4  = BlobModel(NUM_FEATURES, NUM_CLASSES).to(device)

# y_logits = model_4(X_test.to(device))
# y_pred_prob = torch.softmax(y_logits, 1)
# print(y_pred_prob[0])
# print(torch.sum(y_pred_prob[0]))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_4.parameters(), lr=0.1)

torch.manual_seed(RANDOM_SEED)

X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

for epoch in range(100):
    model_4.train()

    y_logits = model_4(X_train)
    y_pred_prob = torch.softmax(y_logits, 1).argmax(1)
    # torch.argmax

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_pred_prob)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_test)
        test_pred_prob = torch.softmax(test_logits, 1).argmax(1)

        test_loss = loss_fn(test_logits, y_test)
        test_acc =  accuracy_fn(y_test, test_pred_prob)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}, Acc: {acc}, Test Loss: {test_loss.item()}, Test Acc: {test_acc}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_test, y_test)
plt.show()