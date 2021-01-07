import torch 
import torch.nn as nn
import numpy as np 

def conv_block(in_channels, latent_dim):
    bn = nn.BatchNorm2d(latent_dim)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, latent_dim, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Conv4_DTN(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, classes=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        #self.latent_dim = 1600
        
        self.fc = MLP(in_dim=1600,z_dim=512,out_dim=64)
        self.latent_dim = 64
        
        # parametric distance
        self.parametric_dist_NN = MLP(in_dim=self.latent_dim*2, out_dim=1)

        # data hallucinate
        self.hallucinate_NN = MLP(in_dim=self.latent_dim+10, z_dim=512,
                              out_dim=self.latent_dim)
        
        # classify
        self.classifier = MLP(in_dim=self.latent_dim, z_dim=512, out_dim=classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) ###
        return x

    def classify(self, x):
        return self.classifier(x)
    
    def weight_norm(self):
        for layer in list(self.classifier.children()):
            if type(layer) is nn.Linear:
                w = layer.weight.data 
                norm = w.norm(p=2, dim=1, keepdim=True)
                layer.weight.data = w.div(norm.expand_as(w))

    def distance(self, a, b): #a(query*ways,z) b(shots*ways,z)
        n = a.size(0)
        m = b.size(0)
        a = a.unsqueeze(1).expand(n,m,-1).contiguous().view(n*m,-1)
        b = b.unsqueeze(0).expand(n,m,-1).contiguous().view(n*m,-1)
        c = torch.cat((a,b), dim=-1)
        logits = self.parametric_dist_NN(c).view(n,m)
        return -logits 

    def hallucinate(self, proto, m): #proto(ways,z)
        n_ways,z_dim = proto.shape

        noise = torch.FloatTensor(np.random.normal(size=(m,n_ways,10))).to(proto.device)
        proto = proto.unsqueeze(0).expand(m,n_ways,z_dim)

        data = torch.cat((noise,proto),dim=2)
        hallu_data = self.hallucinate_NN(data) 
        return hallu_data #(m,ways,z_dim) 

class MLP(nn.Module):
    def __init__(self, in_dim=1600, z_dim=512, out_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(in_dim, z_dim),
                nn.Dropout(0.5),
                nn.ReLU(True),
                nn.Linear(z_dim, out_dim)
            )
    def forward(self, x):
        return self.fc(x)
    
