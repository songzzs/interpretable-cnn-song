# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:00:38 2020

@author: Administrator
"""

from torchvision import models
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

'''定义网络模型'''

class VGG_interpretable_gradcam2(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_interpretable_gradcam2, self).__init__()
        self.num_classes = num_classes
        self.base_model = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-1])
        M = self.base_model[-2].out_channels
        self.add_conv = nn.Conv2d(in_channels=M, out_channels=M,
                                  kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(M)
        
        self.classifier = nn.Sequential(*list(models.vgg16(num_classes=self.num_classes).classifier.children()))
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
        x = self.norm(x)
        x = self.avgpool(x)
        
        if cam:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            x1 = 1
            loss_1 = 1
            x2 = 1
            loss_2 = 1
        
        else:
            rx = x * att
            x1 = self.get_masked_output(rx)
            x = self.add_conv(x1.half())
            #x = self.norm(x)
            x2 = self.get_masked_output(x)
            self.featuremap1 = x2.detach()
            x = x.view(x2.size(0), -1)
            x = self.classifier(x)
    
            # compute local loss:
            loss_1 = self.compute_local_loss(x1)
            loss_2 = self.compute_local_loss(x2)

        return x, x1, x2, loss_1, loss_2

class VGG_interpretable_gradcam(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_interpretable_gradcam, self).__init__()
        self.num_classes = num_classes
        self.base_model = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-1])
        M = self.base_model[-2].out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.norm = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self.classifier = nn.Sequential(*list(models.vgg16(num_classes=self.num_classes).classifier.children()))
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
        x = self.avgpool(x)
        
        if cam:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            x1 = 1
            loss_1 = 1
        
        else:
            rx = x * att
            x1 = self.get_masked_output(rx)
            self.featuremap1 = x1.detach()
            x = x1
            x = self.maxpool(x)
            #self.featuremap2 = x.detach()
            x = x.view(x.size(0), -1)
            x = self.fc(x.half())
            #x = self.classifier(x.half())
    
            # compute local loss:
            loss_1 = self.compute_local_loss(x1)

        return x, x1, loss_1

class VGG_gradcam(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_gradcam, self).__init__()
        self.num_classes = num_classes
        self.base_model = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-1])
        M = self.base_model[-2].out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
        self.classifier = nn.Sequential(*list(models.vgg16(num_classes=self.num_classes).classifier.children()))
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
        x = self.avgpool(x)
        
        if cam:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        
        else:
            rx = x * att
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
    

        return x


class VGG_interpretable_atten(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_interpretable_atten, self).__init__()
        self.num_classes = num_classes
        self.base_model = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-1])
        M = self.base_model[-2].out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.att_conv   = nn.Conv2d(M, num_classes, kernel_size=1, padding=0,
                               bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0,
                               bias=False)
        self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1,
                               bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.laynorm = nn.LayerNorm(normalized_shape=[1,7,7], eps=0, elementwise_affine=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.norm = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self.classifier = nn.Sequential(*list(models.vgg16(num_classes=self.num_classes).classifier.children()))
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

    def forward(self, x, train=True):
        x = self.base_model(x)
        x = self.avgpool(x)
        
        ax = self.relu(self.bn_att2(self.att_conv(x)))
        ax1 = self.bn_att3(self.att_conv3(ax))
        ax2 = self.laynorm(ax1)
        self.att = self.sigmoid(ax2)
        ax = self.att_conv2(ax)
        ax = self.att_gap(ax)
        ax = ax.view(ax.size(0), -1)
        
        rx = x * self.att
        x1 = self.get_masked_output(rx)
        #x1 = self.norm(x1.half())
        x = x1
        self.featuremap1 = x.detach()
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x.half())
        #x = self.classifier(x.half())

        # compute local loss:
        loss_1 = self.compute_local_loss(x1)
        #loss_2 = self.compute_local_loss(x2)

        return ax, x, rx, loss_1
    
class VGG_atten(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_atten, self).__init__()
        self.num_classes = num_classes
        self.base_model = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-1])
        M = self.base_model[-2].out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.att_conv   = nn.Conv2d(M, num_classes, kernel_size=1, padding=0,
                               bias=False)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        self.att_conv2  = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0,
                               bias=False)
        self.att_conv3  = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1,
                               bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.laynorm = nn.LayerNorm(normalized_shape=[1,7,7], eps=0, elementwise_affine=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        
        self.classifier = nn.Sequential(*list(models.vgg16(num_classes=self.num_classes).classifier.children()))
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

    def forward(self, x, train=True):
        x = self.base_model(x)
        x = self.avgpool(x)
        
        ax = self.relu(self.bn_att2(self.att_conv(x)))
        ax1 = self.bn_att3(self.att_conv3(ax))
        ax2 = self.laynorm(ax1)
        self.att = self.sigmoid(ax2)
        ax = self.att_conv2(ax)
        ax = self.att_gap(ax)
        ax = ax.view(ax.size(0), -1)
        
        rx = x * self.att
        rx = self.maxpool(rx)
        x = rx.view(rx.size(0), -1)
        x = self.fc(x)
        #x = self.classifier(x)

        return ax, x

'''class VGG_interpretable(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_interpretable, self).__init__()
        self.num_classes = num_classes
        self.base_model = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-1])
        M = self.base_model[-2].out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        #self.add_conv = nn.Conv2d(in_channels=M, out_channels=M,
                                  #kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Sequential(*list(models.vgg16(num_classes=self.num_classes).classifier.children()))
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

    def forward(self, x, train=True):
        x = self.base_model(x)
        x = self.avgpool(x)
        x1 = self.get_masked_output(x)
        self.featuremap1 = x1.detach()
        #x = self.add_conv(x1)
        #x2 = self.get_masked_output(x)
        #x = F.max_pool2d(x1, 2, 2)
        x = x.view(x1.size(0), -1)
        x = self.classifier(x.half())

        # compute local loss:
        loss_1 = self.compute_local_loss(x1)
        #loss_2 = self.compute_local_loss(x2)

        return x, x1, loss_1'''
    
class VGG_interpretable(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_interpretable, self).__init__()
        self.num_classes = num_classes
        self.base_model = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-1])
        M = self.base_model[-2].out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.add_conv = nn.Conv2d(in_channels=M, out_channels=M,
                                  kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        #self.classifier = nn.Sequential(*list(models.vgg16(num_classes=self.num_classes).classifier.children()))
        # create templates for all filters
        self.out_size = 7

        mus = torch.FloatTensor([[i, j] for i in range(self.out_size) for j in range(self.out_size)])
        templates = torch.zeros(mus.size(0), self.out_size, self.out_size)

        n_square = self.out_size * self.out_size

        tau = 0.5 / n_square
        #tau = 1
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
        p_temp = torch.log(p_x_T/p_x[:, :, None])
        #mask = (p_temp == float('-inf'))
        #p_temp = p_temp.data.masked_fill(mask, 0)
        p_x_T_log = (p_x_T *p_temp ).sum(1)
        #mask = (p_x_T_log == float('-inf'))
        #p_x_T_log = p_x_T_log.data.masked_fill(mask, 0)
        loss = - (self.p_T[None, :] * p_x_T_log).sum(-1)
        #mask = torch.isnan(loss)
        #loss = Variable(loss.data.masked_fill(mask, 0),requires_grad=True)
        #loss.requires_grad = True
        return loss

    def forward(self, x, train=True):
        x = self.base_model(x)
        ax1 = self.avgpool(x)
        x1 = self.get_masked_output(ax1)
        self.featuremap1 = x1.detach()
        #ax2 = self.add_conv(x1.half())
        #x2 = self.get_masked_output(ax2)
        x = x1
        #x = F.max_pool2d(x2, 2, 2)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x.half())
        #x = self.classifier(x.half())

        # compute local loss:
        loss_1 = self.compute_local_loss(x1)
        #loss_2 = self.compute_local_loss(x2)

        return x, x1, loss_1#, x2, loss_2

class VGGNet(nn.Module):
    def __init__(self, num_classes=2):	   #num_classes，此处为 二分类值为2
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)   #从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()	#将分类层置空，下面将改变我们的分类层
        #self.features = net		#保留VGG16的特征层
        #self.base_model = net
        self.base_model = nn.Sequential(*list(models.vgg16(pretrained=True).features.children()))
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.classifier = nn.Sequential(    #定义自己的分类层
                nn.Linear(512 * 7 * 7, 512),  #512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.maxpool(x)
        #self.featuremap1 = x
        x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        x = self.fc(x)
        return x