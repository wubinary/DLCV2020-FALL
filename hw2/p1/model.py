import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, vgg16, vgg19

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.features = vgg16(pretrained=True).features
        self.fc = nn.Sequential(
                        nn.Linear(128*8*8, 512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(512, num_classes),
                    )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

if __name__=='__main__':
    m = Model(50)
    print(m)

