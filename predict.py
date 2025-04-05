# 该模块主要是为了预测分类，输入一个图像得到一个类别
'''
# Part1 引入相关的模型
'''
import torch
from dataset import Mnist_dataset
import matplotlib.pyplot as plt

'''
# part2 下载模型
'''
net = torch.load('VIT_eopch_0.pt')
net.eval()
data_cs = Mnist_dataset(is_tran=False)

'''
# Part3 开始测试
'''
if __name__ == '__main__':
    img, label = data_cs[1]
    label_predict = net(img.unsqueeze(0))
    label_predict = torch.argmax(label_predict)
    if label_predict == label:
        print('真实的标签为{},预测的标签为{}，预测正确'.format(label, label_predict))
    else:
        print('真实的标签为{},预测的标签为{}，预测错误'.format(label, label_predict))
    # 开始绘制图像
    plt.imshow(img.permute(2,1,0))
    plt.show()

