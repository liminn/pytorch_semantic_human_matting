import torch
from torch import nn
from torch.nn import functional as F

from model import extractors

"""
from:
    https://github.com/Lextal/pspnet-pytorch
Details
    This is a slightly different version - 
    instead of direct 8x upsampling at the end I use three consequitive upsamplings for stability.
"""

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)

class PSPNet(nn.Module):
    def __init__(self, n_classes=3, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50',
                 pretrained=True):
        super().__init__()
        # 输出为dilated ResNet的不含顶部的特征，即最后两层为dilated conv，特征图为1/8
        self.feats = getattr(extractors, backend)(pretrained)
        # 输出为经PSP Module后，1/8,通道数为1024的特征
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        # 区别于原文，逐步进行上采样
        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)
        self.drop_2 = nn.Dropout2d(p=0.15)

        ### 区别于源代码：由于外部要用nn.CrossEntropyLoss()，不需要先进行softmax，故去除
        # self.final = nn.Sequential(
        #     nn.Conv2d(64, n_classes, kernel_size=1),
        #     nn.LogSoftmax()    # 源代码训练时损失函数为nn.NLLLoss2d()，需先继续LogSoftmax
        # )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(deep_features_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_classes)
        # )
        
    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)
        
        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        
        #auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
        
        #return self.final(p),self.classifier(auxiliary)
        return self.final(p)

