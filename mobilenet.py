#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:55:57 2020

@author: song
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F


class MobileNet(nn.Module):
    def __init__(self,num_classes=10):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):    # 第一层传统的卷积：conv3*3+BN+ReLU
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):      # 其它层的depthwise convolution：conv3*3+BN+ReLU+conv1*1+BN+ReLU
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2),   # 第一层传统的卷积
            conv_dw( 32,  64, 1),   # 其它层depthwise convolution
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            #nn.AvgPool2d(7),
        )
        self.fc0 = nn.Linear(1024*7*7,1024)
        self.fc = nn.Linear(1024, num_classes)   # 全连接层

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024*7*7)
        x = self.fc0(x)
        x = self.fc(x)
        return x
    
class Mobile_interpretable_gradcam(nn.Module):
    def __init__(self, num_classes=10):
        super(Mobile_interpretable_gradcam, self).__init__()
        self.num_classes = num_classes
        self.base_model = nn.Sequential(*list(MobileNet().children())[:-2])
        M = 1024 #self.base_model[-2].out_channels
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
        #self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.fc0 = nn.Linear(1024*7*7,1024)
        self.classifier = nn.Linear(1024,num_classes)
        # create templates for all filters
        self.out_size = 7

        mus = torch.FloatTensor([[i, j] for i in range(self.out_size) for j in range(self.out_size)])
        templates = torch.zeros(mus.size(0), self.out_size, self.out_size)

        n_square = self.out_size * self.out_size

        tau = 0.5 / n_square
        alpha = n_square / (1 + n_square)
        beta = 4

        for k in range(templates.size(0)):
            for i in range(self.out_size):
                for j in range(self.out_size):
                    if k < templates.size(0) - 1:  # positive templates
                        norm = (torch.FloatTensor([i, j]) - mus[k]).norm(1, -1)
                        out = tau * torch.clamp(1 - beta * norm / self.out_size, min=-1)
                        templates[k, i, j] = float(out)

        self.templates_f = Variable(templates, requires_grad=False).cuda(0)
        neg_template = -tau * torch.ones(1, self.out_size, self.out_size)
        templates = torch.cat([templates, neg_template], 0)
        self.templates_b = Variable(templates, requires_grad=False).cuda(0)

        p_T = [alpha / n_square for _ in range(n_square)]
        p_T.append(1 - alpha)
        self.p_T = Variable(torch.FloatTensor(p_T), requires_grad=False).cuda(0)

    def get_masked_output(self, x):
        # choose template that maximize activation and return x_masked
        indices = F.max_pool2d(x, self.out_size, return_indices=True)[1].squeeze()
        selected_templates = torch.stack([self.templates_f[i] for i in indices], 0)
        x_masked = F.relu(x * selected_templates)
        return x_masked

    def compute_local_loss(self, x):
        x = x.permute(1, 0, 2, 3)
        exp_tr_x_T = (x[:, :, None, :, :] * self.templates_b[None, None, :, :, :]).sum(-1).sum(-1).exp()
        Z_T = exp_tr_x_T.sum(1, keepdim=True)
        p_x_T = exp_tr_x_T / Z_T

        p_x = (self.p_T[None, None, :] * p_x_T).sum(-1)
        p_x_T_log = (p_x_T * torch.log(p_x_T/p_x[:, :, None])).sum(1)
        loss = - (self.p_T[None, :] * p_x_T_log).sum(-1)
        return loss

    def forward(self, x, att, cam=True):
        x = self.base_model(x)
        
        if cam:
            #x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc0(x)
            x = self.classifier(x)
            x1 = 1
            loss_1 = 1
        
        else:
            rx = x * att
            x1 = self.get_masked_output(rx)
            self.featuremap1 = x1.detach()
            #x1 = self.avgpool(x1)
            x = x1.view(x1.size(0), -1)
            x = self.fc0(x.half())
            x = self.classifier(x)
    
            # compute local loss:
            loss_1 = self.compute_local_loss(x1)

        return x, x1, loss_1