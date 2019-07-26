from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import math

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):

        super(Bottleneck, self).__init__()

        wide_factor = 1

        conv1_in = in_channels
        conv1_out = math.floor(out_channels * wide_factor)

        conv2_in = math.floor(out_channels * wide_factor)
        conv2_out = math.floor(out_channels * wide_factor)

        conv3_in = math.floor(out_channels * wide_factor)
        conv3_out = out_channels * 4

        self.conv_conv1 = nn.Conv2d(conv1_in, conv1_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_bn1 = nn.BatchNorm2d(conv1_out)
        self.conv_conv2 = nn.Conv2d(conv2_in, conv2_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_bn2 = nn.BatchNorm2d(conv2_out)
        self.conv_conv3 = nn.Conv2d(conv3_in, conv3_out, kernel_size=1, stride=1 ,padding=0, bias=False)
        self.bn_bn3 = nn.BatchNorm2d(conv3_out)

        self.Pool = nn.AvgPool2d(7,7)
        self.fc_spatial = nn.Sequential()
        if out_channels==64:
            self.fc_spatial.add_module('fc_spatial', nn.Conv2d(conv3_out, conv3_out, kernel_size=8, stride=8, groups=conv3_out, bias=False))
            self.fc_spatial.add_module('bn_spatial', nn.BatchNorm2d(conv3_out))
        elif out_channels==128:
            self.fc_spatial.add_module('fc_spatial', nn.Conv2d(conv3_out, conv3_out, kernel_size=4, stride=4, groups=conv3_out, bias=False))
            self.fc_spatial.add_module('bn_spatial', nn.BatchNorm2d(conv3_out))
        elif out_channels==256:
            self.fc_spatial.add_module('fc_spatial', nn.Conv2d(conv3_out, conv3_out, kernel_size=2, stride=2, groups=conv3_out, bias=False))
            self.fc_spatial.add_module('bn_spatial', nn.BatchNorm2d(conv3_out))

        self.fc_reduction = nn.Linear(in_features=conv3_out, out_features=conv3_out//16)
        self.fc_extention = nn.Linear(in_features=conv3_out//16 , out_features=conv3_out)
        self.sigmoid = nn.Sigmoid()

        self.shortcut = nn.Sequential()
        if (in_channels != out_channels * 4) or stride != 1:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels*4))


    def forward(self, x):

        res = self.conv_conv1.forward(x)
        res = F.relu(self.bn_bn1.forward(res), inplace=True)

        res = self.conv_conv2.forward(res)
        res = F.relu(self.bn_bn2.forward(res), inplace=True)

        res = self.conv_conv3.forward(res)
        res = self.bn_bn3.forward(res)

        se_out = self.Pool(res)
        se_out = self.fc_spatial(se_out)
        se_out = F.relu(se_out,inplace=True)
        se_out = se_out.view(se_out.size(0), -1)
        se_out = self.fc_reduction(se_out)
        se_out = F.relu(se_out, inplace=True)
        se_out = self.fc_extention(se_out)
        se_out = self.sigmoid(se_out)
        se_out = se_out.view(se_out.size(0),se_out.size(1),1,1)# batch_size x channel x 1 x 1

        res = se_out*res

        shtcut = self.shortcut.forward(x)

        return F.relu(res + shtcut, inplace=True)

class RGSE_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 1000):
        self.inplanes = 64
        super(RGSE_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer('layer1', block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer('layer2', block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer('layer3', block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer('layer4', block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, name, block, planes, blocks, stride):#out_channel = planes

        layers = nn.Sequential()
        for block_idx in range(blocks):
            name_ = '%s_block_%d' % (name, block_idx)
            if block_idx == 0:
                layers.add_module(name_, block(self.inplanes, planes, stride))
                self.inplanes = planes * block.expansion
            else:
                layers.add_module(name_, block(self.inplanes, planes, 1))
        return layers

    def forward(self, x):
        x = self.conv1.forward(x)
        x = F.relu(self.bn1.forward(x), inplace=True)
        x = self.maxpool(x)

        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        x = self.layer4.forward(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50_rgse():
    return RGSE_ResNet(Bottleneck,[3,4,6,3])

def resnet101_rgse():
    return RGSE_ResNet(Bottleneck,[3,4,23,3])

def resnet152_rgse():
    return RGSE_ResNet(Bottleneck,[3,8,36,3])