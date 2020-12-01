from models.reverse_grad_layer import RevGrad

import torchvision.models as ms 

import torch
import torch.nn as nn

def init_weights(model, normal=True):
    if normal:
        init_func = nn.init.xavier_normal_
        init_zero = nn.init.zeros_
    else:
        init_func = nn.init.xavier_uniform_
        init_zero = nn.init.zeros_

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init_func(m.weight.data)
            if m.bias is not None:
                init_zero(m.bias.data)
        elif isinstance(m, nn.Linear):
            init_func(m.weight.data)
            if m.bias is not None:
                init_zero(m.bias.data)

####################################################
##################  DANN Model  ####################
'''
class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.apply(init_weights)
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x
'''
class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
                *list(ms.resnet18(pretrained=False).children())[:-1]
            )
        
    def forward(self, x):
        bsize = x.size(0)
        x = self.conv(x)
        return x.view(bsize,-1)

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y
