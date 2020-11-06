from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import glob
import torch
import numpy as np


class p1_Dataset(Dataset):
    def __init__(self, path=None, transform=None):
        assert path is not None
        self.path = path
        self.data = [(img_path,i) for i in range(50) for img_path in glob.glob(f'{self.path}/{i}_*.png')]
        self.transform = transform
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img,lbl = Image.open(self.data[idx][0]).convert('RGB'), self.data[idx][1]
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl
    
class p1_Dataset_test(Dataset):
    def __init__(self, path=None, transform=None):
        assert path is not None
        self.path = path
        self.data = [img_path for img_path in sorted(glob.glob(f'{self.path}/*.png'))]
        self.transform = transform
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.data[idx]
