from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import glob
import torch
import numpy as np

#from utils import xx

class Face_Dataset(Dataset):
    def __init__(self, path=None, transform=None):
        assert path is not None
        self.path = path 
        self.data = glob.glob(path+"/*.png")
        self.transform = transform
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img)

        img_in = img_out = img
        
        return img_in#, img_out
    
class Face_Dataset_Test(Dataset):
    def __init__(self, in_path=None, out_path=None, transform=None):
        assert in_path is not None
        assert out_path is not None
        self.data = glob.glob(in_path+"/*.png")
        self.out_paths = [p.replace(in_path, out_path) for p in self.data]
        self.transform = transform
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img)

        img_in = img_out = img
        path = self.out_paths[idx]
        
        return img_in, path


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    ds = Face_Dataset('../hw3_data/face/train', transform)
    
    
