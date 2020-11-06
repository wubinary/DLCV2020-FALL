import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, vgg16, vgg19 

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class VGG16_FCN8s(nn.Module):
    def __init__(self, n_classes=7):
        super(VGG16_FCN8s, self).__init__()
        
        ## VGG16
        self.features = vgg16(pretrained=True).features
        self.features[0].padding = [100,100]
        #self.features._modules['0'].padding = 100
        #for param in self.features.parameters():
        #    param.requires_grad = False
        
        ## FCN8s
        self.fcn = nn.Sequential(
                nn.Conv2d(512, 4096, 7),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d()
            )
        
        self.score_pool3 = nn.Conv2d(256, n_classes, 1)
        self.score_pool4 = nn.Conv2d(512, n_classes, 1)
        self.score_fr = nn.Conv2d(4096, n_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_classes, n_classes, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        w, h = x.size()[2:]
        
        y = x
        for l in self.features[:17]:
            y = l(y)
        pool3 = y

        for l in self.features[17:24]:
            y = l(y)
        pool4 = y
        
        for l in self.features[24:]:
            y = l(y)

        y = self.fcn(y)
        y = self.score_fr(y)
        y = self.upscore2(y)
        upscore2 = y #1/16

        y = self.score_pool4(pool4)
        y = y[:,:,5:5+upscore2.size()[2], 5:5+upscore2.size()[3]]
        score_pool4c = y #1/16
        
        y = upscore2 + score_pool4c
        y = self.upscore_pool4(y)
        upscore_pool4 = y #1/8

        y = self.score_pool3(pool3)
        y = y[:,:,9:9+upscore_pool4.size()[2],9:9+upscore_pool4.size()[3]]
        score_pool3c = y #1/8

        y = upscore_pool4 + score_pool3c 
        y = self.upscore8(y)
        y = y[:,:,31:31+x.size()[2],31:31+x.size()[3]].contiguous()
        
        return y 

class VGG16_FCN32s(nn.Module):
    def __init__(self, n_classes=7):
        super(VGG16_FCN32s, self).__init__()
        
        ## VGG16
        self.features = vgg16(pretrained=True).features 
        self.features[0].padding = [100,100]
        #self.features._modules['0'].padding = 100
        #for param in self.features.parameters():
        #    param.requires_grad = False
       
        ## FCN32s
        self.fcn32 = nn.Sequential(
                nn.Conv2d(512, 4096, 7),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(4096, n_classes, 1)
            )        

        self.upscore = nn.ConvTranspose2d(n_classes, n_classes, 64, stride=32,
                                          bias=False)  

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        w, h = x.size()[2:]

        y = self.features(x)
        y = self.fcn32(y)
        y = self.upscore(y)
        y = y[:,:, 32:32 + x.size()[2], 32:32 + x.size()[3]].contiguous()
        
        return y

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


if __name__=='__main__':
    model = VGG16_FCN8s(7)
    print(model)

