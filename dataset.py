# 该模块主要是为了嵌入获取相关的数据，得到dataloader等等,只不过把数据集的获取设计成了一个类
import torch
import torchvision
from torchvision import transforms
from torch import nn
# 总是容易忘记这个data的引用
import torch.utils.data as data


class Mnist_dataset(data.Dataset):
    def __init__(self, is_tran):
        super().__init__()
        self.transform_action = transforms.Compose([
            transforms.ToTensor()
        ])
        self.Mnist = torchvision.datasets.MNIST(root='./Mnist_data', train=is_tran, transform=self.transform_action,
                                                download=True)

    def __len__(self):
        return len(self.Mnist)

    def __getitem__(self, index):
        return self.Mnist[index][0], self.Mnist[index][1]  # 返回对应的图像和label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.figure(figsize=(45, 45))
    mnist = Mnist_dataset(is_tran=True)
    img, label = mnist[0]
    # 因为Totensor，会把数据进行归一化，并且把pillow(imag_size,imag_size,channel)转化为(channel,imag_size,imag_size)
    # 为了能够画图，需要把channel换回来
    plt.imshow(img.permute(2, 1, 0))
    plt.title(f"Label: {label}")
    plt.show()
