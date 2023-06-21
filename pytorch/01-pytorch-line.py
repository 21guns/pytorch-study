#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt
'''
    线性回归
'''
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X =  torch.arange(start,end,step).unsqueeze(1)
Y = weight*X + bias
# print(X)
# print(Y)
train_split = int(0.8 * len(X))
train_X = X[:train_split]
train_Y = Y[:train_split]
test_X = X[train_split:]
test_Y = Y[train_split:]

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.randn(1), requires_grad=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight*x + self.bias

class LinearRegressionV2(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionV2, self).__init__()
        self.linea_layer = torch.nn.Linear(1,1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linea_layer(x)
    
torch.manual_seed(42)
model_0 = LinearRegression()
# print(list(model_0.parameters()))

with torch.inference_mode():
    y_0 = model_0(train_X)
# print(y_0)
# print(train_X)

loss_func = torch.nn.L1Loss()

optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

torch.manual_seed(42)

epchs = 200

train_loss_values=[]
test_loss_values=[]
epoch_count = []

for epoch in range(epchs):
    model_0.train()
    y_1 = model_0(train_X)
    loss = loss_func(y_1, train_Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        y_2 = model_0(test_X)
        test_loss = loss_func(y_2, test_Y)
    
    if epoch % 10 == 0:
        # print(f'epoch {epoch} train loss: {loss.item():.3f} test loss: {test_loss.item():.3f}')
        train_loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())
        epoch_count.append(epoch)

# Plot the loss curves
# plt.plot(epoch_count, train_loss_values, label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()

model_0.eval()
with torch.inference_mode():
    y_3 = model_0(X)
    print(y_3)