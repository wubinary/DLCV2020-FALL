import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

def get_transform():
    size=84
    train_trans = transforms.Compose([
            transforms.RandomResizedCrop(size,scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
                ],p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    valid_trans = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return train_trans, valid_trans 

class MiniImageNet_Dataset(Dataset):
    def __init__(self, path, transform):
        self.path = path.rstrip('/')
        
        df = pd.read_csv(self.path+'.csv')
        df['label2'] = df['label'].astype('category').cat.codes
        
        self.transform = transform
        self.data = [os.path.join(self.path,fname) for fname in df['filename']]
        self.label = df['label2'].tolist()
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
        
class CategoriesSampler():
    def __init__(self, label, n_batch, n_ways=5, n_shot=1):
        self.n_batch = n_batch
        self.n_ways = n_ways
        self.n_shot = n_shot
        
        self.category_idxs = []
        
        label = np.array(label)
        for l in set(label):
            idxs = np.argwhere(label==l).reshape(-1)
            self.category_idxs.append(torch.from_numpy(idxs))
    
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.category_idxs))[:self.n_ways]
            for c in classes:
                l = self.category_idxs[c]
                shot_idxs = torch.randperm(len(l))[:self.n_shot]
                batch.append(l[shot_idxs])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch #(n_shot, n_way).reshpae(-1)
