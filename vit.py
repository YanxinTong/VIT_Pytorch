# 该模块主要是为了实现Vit模块，对图像进行分类

'''
#Part1 引入相关库函数
'''
import torch
from torch import nn

'''
# Part2 设计一个Vit类，实现输入图像输出对应的概率向量
'''


class VIT(nn.Module):
    def __init__(self, img_channel, imag_size, patch_size, emd_size, num_kind):
        super().__init__()
        # 首先对图像进行一个patch 化,然后进行卷积，之后展平(可以直接resize就行，会按顺序从第一行到第n行)。
        self.patch_size = patch_size
        self.patch_count = imag_size // patch_size
        # 因为保证是整数，所以我们需要先对每个patch块(batch,channel,patch_size,patch_size)进行卷积为(batch,channel*patchsize**2,1,1),也就
        # 相当于进行拉伸,所以对于每个图像来说，(batch,channel,img_size,img_size)->(batch,channel*patchsize**2,patch_count,patch_count)
        self.conv1 = nn.Conv2d(in_channels=img_channel, out_channels=img_channel * patch_size ** 2,
                               kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)

        # 主要是因为语言其实是没有通道的概念的，所以为了把图像和通道联合在一起，把通道作为词嵌入(所以要移动到最后)，然后把count,count，变成count*count.也就是变成(batch,seq_len,emd_size)
        self.linear1 = nn.Linear(in_features=img_channel * patch_size ** 2,
                                 out_features=emd_size)  # ((batch,patch_count*patch_count,emd_size))

        # 因为需要利用[CLS]分类头(相当于融合了整个句子的特征，作为整个句子的表示)做分类，所以需要在每个图像前面插入一个分类头。
        self.class_head = nn.Parameter(torch.rand(size=(1, 1, emd_size)))  # (batch,1,emd_size)
        # 然后就相当于得到了和当时transformer一样的输入形式
        # 所以先对嵌入的维度，进行添加位置嵌入，然后输入transformer的编码器
        self.pos_emd = nn.Parameter(torch.rand(size=(1, self.patch_count ** 2 + 1, emd_size)))
        # 设置transformer的编码器
        self.transformerencoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=emd_size, nhead=2, batch_first=True), num_layers=3)
        # 得到和初始一样大小的输出(batch,self.patch_count**2+1,emd_size),然后把emd_size,变为10，然后取第一个作为分类标准
        self.linear2 = nn.Linear(emd_size, num_kind)

    def forward(self, x):  # (batch,channel,imag_size,imag_size)
        x = self.conv1(x)  # (batch,channel*patchsize**2,patch_count,patch_count)
        # 交换通道
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size()[0], -1, x.size()[-1])
        x = self.linear1(x)  # ((batch,patch_count*patch_count,emd_size))
        class_head = self.class_head.expand(x.size()[0], 1, x.size()[-1])

        x = torch.cat((class_head, x), dim=1)  # ((batch,patch_count*patch_count+1,emd_size))
        x = x + self.pos_emd.expand(x.size()[0], -1, -1)
        x = self.transformerencoder(x)
        x = self.linear2(x)  # # ((batch,patch_count*patch_count+1,num_kind))
        return x[:, 0, :]


# 测试
if __name__ == '__main__':
    x = torch.rand(5, 1, 28, 28)
    vit = VIT(img_channel=1, imag_size=28, patch_size=7, emd_size=36, num_kind=10)
    y = vit(x)
    print(y.shape)
