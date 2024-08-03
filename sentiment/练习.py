from torch.utils.data import DataLoader, dataset

DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
 )
"""
dataset:是数据集
batch_size:是指一次迭代中使用的训练样本数。通常我们将数据分成训练集和测试集，并且我们可能有不同的批量大小。
shuffle:是传递给 DataLoader 类的另一个参数。该参数采用布尔值（真/假）。如果 shuffle 设置为 True，则所有样本都被打乱并分批加载。否则，它们会被一个接一个地发送，而不会进行任何洗牌。
num_workers:允许多处理来增加同时运行的进程数
collate_fn：合并数据集
pin_memory:锁页内存：将张量固定在内存中
"""
# Import MNIST
from torchvision.datasets import MNIST

# Download and Save MNIST
data_train = MNIST('~/mnist_data', train=True, download=True)

# Print Data
# print(data_train)
# print(data_train[12])

#Dataset MNIST Number of datapoints: 60000 Root location: /Users/viharkurama/mnist_data Split: Train (<PIL.Image.Image image mode=L size=28x28 at 0x11164A100>, 3)
import matplotlib.pyplot as plt

random_image = data_train[0][0]
random_image_label = data_train[0][1]

# Print the Image using Matplotlib
plt.imshow(random_image)
print("The label of the image is:", random_image_label)

import torch
from torchvision import transforms

data_train = torch.utils.data.DataLoader(
    MNIST(
          '~/mnist_data', train=True, download=True,
          transform = transforms.Compose([
              transforms.ToTensor()
          ])),
          batch_size=64,
          shuffle=True
          )

# for batch_idx, samples in enumerate(data_train):
      # print(batch_idx, samples)




