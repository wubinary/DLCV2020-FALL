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
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

# Generator 
class Generator(nn.Module):
    def __init__(self, in_dim=100, dim=64, c=3):
        super(Generator, self).__init__()
        '''
        input (N, in_dim)
        output (N, 3, 64, 64)
        '''
        def deconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                        nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                            padding=2, output_padding=1, bias=False),
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU()
                    )
        self.l1 = nn.Sequential(
                nn.Linear(in_dim, dim*8*4*4, bias=False),
                nn.BatchNorm1d(dim*8*4*4),
                nn.ReLU()
            )
        self.l2_5 = nn.Sequential(
                deconv_bn_relu(dim*8, dim*4),
                deconv_bn_relu(dim*4, dim*2),
                deconv_bn_relu(dim*2, dim),
                nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
                nn.Tanh()
            )
        
        self.apply(init_weights)

    def forward(self, input):
        y = self.l1(input)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, c=3, dim=64):
        super(Discriminator, self).__init__()
        '''
        input (N, 3, 64, 64)
        output (N, )
        '''
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                        nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                        nn.BatchNorm2d(out_dim),
                        nn.LeakyReLU(0.2)
                    )
        self.main = nn.Sequential(
                nn.Conv2d(c, dim, 5, 2, 2),
                nn.LeakyReLU(0.2),
                conv_bn_lrelu(dim, dim*2),
                conv_bn_lrelu(dim*2, dim*4),
                conv_bn_lrelu(dim*4, dim*8),
                nn.Conv2d(dim*8, 1, 4),
                nn.Sigmoid()
            )
        
        self.apply(init_weights)

    def forward(self, input):
        return self.main(input)
    
