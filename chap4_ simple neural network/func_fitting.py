import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定义一个简单的二次函数 f(x) = x^2


def quadratic_function(x):
    return 2*x**2+3*x+4


# 创建训练数据
x_train = np.linspace(-5, 5, 100).reshape(-1, 1)
y_train = quadratic_function(x_train)

# 转换为 PyTorch 的 Tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# 定义一个两层的ReLU网络


class SimpleReLUModel(nn.Module):
    def __init__(self):
        super(SimpleReLUModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 输入层到隐藏层
        self.fc2 = nn.Linear(64, 1)  # 隐藏层到输出层
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 创建模型实例
model = SimpleReLUModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 将训练结果转换为 numpy 数组
x_test = np.linspace(-5, 5, 100).reshape(-1, 1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
with torch.no_grad():
    y_pred_tensor = model(x_test_tensor)
y_pred = y_pred_tensor.numpy()

# 可视化训练结果
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, label='Input data')
plt.plot(x_test, y_pred, 'r-', label='Output')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
