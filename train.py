# 训练模型，该模块主要是为了实现对于模型的训练，
'''
# Part1 引入相关的库函数
'''

import torch
from torch import nn
from dataset import Mnist_dataset
from vit import VIT
import torch.utils.data as data

'''
初始化一些训练参数
'''
EPOCH = 50
Mnist_train = Mnist_dataset(is_tran=True)
Mnist_dataloader = data.DataLoader(dataset=Mnist_train.Mnist, batch_size=64, shuffle=True)

# 前向传播的模型
net = VIT(img_channel=1, imag_size=28, patch_size=7, emd_size=20, num_kind=10)

# 计算损失函数
loss = nn.CrossEntropyLoss()

# 反向更新参数
lr = 1e-3
optim = torch.optim.Adam(params=net.parameters(), lr=lr)

'''
# 开始训练
'''
# net.train() # 设置为训练模式

for epoch in range(EPOCH):
    n_iter = 0
    for batch_img,batch_label in Mnist_dataloader:
        # 先进行前向传播
        batch_label_predict=net(batch_img) #

        # 计算损失
        loss_cal=loss(batch_label_predict,batch_label)

        # 清除梯度
        optim.zero_grad()
        # 反向传播
        loss_cal.backward()
        # 更新参数
        optim.step()

        l=loss_cal.item()


        if n_iter%100==0:
            print('此时的epoch为{},iter为{},loss为{}'.format(epoch,n_iter,l))

        n_iter += 1
    if epoch==0:
        # 注意pt文件是保存整个模型及其参数的，pth文件只是保存参数
        torch.save(net,'VIT_eopch_{}.pt'.format(epoch))