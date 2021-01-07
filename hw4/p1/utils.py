import torch 
import torch.nn.functional as F
import numpy as np

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def accuracy_(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

class Averager():
    def __init__(self):
        self.clean()

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    
    def clean(self):
        self.n = 0
        self.v = 0

def distance_metric(method='dot', model=None):
    ''' 
        Args: a(query*ways,z) b(ways,z)
    '''
    def dot(a, b):
        logits = torch.mm(a, b.t())
        return -logits 
    def cosine(a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(0)
        logits = F.cosine_similarity(a,b,dim=2)
        return -logits
    def euclidian(a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = ((a - b)**2).sum(dim=2)
        return -logits
    def pairwise(a, b, p=3):
        a = a.unsqueeze(1)
        b = b.unsqueeze(0)
        logits = F.pairwise_distance(a,b,p=p)
        return -logits 

    if method == 'dot':
        return dot
    elif method == 'cosine':
        return cosine 
    elif method == 'euclidian':
        return euclidian
    elif method == 'pairwise':
        return pairwise
    elif method == 'parametric':
        return model.distance
    else:
        raise NotImplementedError(f"{method} not implemented")

