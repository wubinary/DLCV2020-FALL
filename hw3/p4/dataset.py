#from utils import xx

from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import glob
import torch 
import numpy as np
import pandas as pd 

# https://pytorch-ada.readthedocs.io/en/latest/_modules/ada/datasets/preprocessing.html


class Digits_Dataset(Dataset):
    def __init__(self, path=None, dataset=None, transform=None):
        assert path is not None
        self.path = path.strip('/')
        df = pd.read_csv(self.path+'.csv')

        self.data = df.image_name.tolist()
        self.label = df.label.tolist()
        self.transform = transform#[dataset]
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(f"{self.path}/{self.data[idx]}").convert('RGB') 
        if self.transform is not None:
            img = self.transform(img)
        
        lbl = self.label[idx]
        
        return img, lbl 

    def collate_fn(self, samples):
        img_batch, lbl_batch = [], []
        
        for img,lbl in samples:
            img_batch.append(img.unsqueeze(0))
            lbl_batch.append(lbl)
        
        img_batch = torch.cat(img_batch, dim=0)
        lbl_batch = torch.tensor(lbl_batch)

        return img_batch, lbl_batch 

class Digits_Dataset_Test(Dataset):
    def __init__(self, in_path=None, transform=None):
        assert in_path is not None
        self.in_path = in_path
        
        self.data = glob.glob(in_path+"/*.png")
        self.transform = transform#[dataset]
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img)
                    
        fname = self.data[idx].replace(self.in_path,'').replace('/','')
        
        return img, fname

    def collate_fn(self, samples):
        img_batch, fname_batch = [], []
        
        for img,fname in samples:
            img_batch.append(img.unsqueeze(0))
            fname_batch.append(fname)
        
        img_batch = torch.cat(img_batch, dim=0)

        return img_batch, fname_batch 
        
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    ds = Digits_Dataset('../hw3_data/digits/mnistm/train/', transform)
    a,b = ds.collate_fn([ds[1],ds[3],ds[11]])
    print(a.shape, b.shape)
    
