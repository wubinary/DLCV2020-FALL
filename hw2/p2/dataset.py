from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import glob
import torch
import numpy as np

from utils import mask2label
from mean_iou_evaluate import read_masks

class_map = [(0,255,255),(255,255,0),(255,0,255),(0,255,0),(0,0,255),
             (255,255,255),(0,0,0)]
classes = ['Urban','Agriculture','Rangeland','Forest','Water','Barren','Unknown']

class p2_Dataset(Dataset):
    def __init__(self, path=None, transform=None, valid=False):
        assert path is not None
        self.path = path 
        max_len = 257 if valid else 2000
        self.data = [("{}/{:04}_sat.jpg".format(path,i),
                      "{}/{:04}_mask.png".format(path,i)) 
                    for i in range(max_len)]
        self.transform = transform
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx][0]).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img)

        lbl_img = mask2label(self.data[idx][1])
        lbl_img = torch.tensor(lbl_img).long()
        
        return img, lbl_img

class p2_Dataset_test(Dataset):
    def __init__(self, path=None, transform=None, out_transform=None):
        assert path is not None
        self.path = path 
        self.data = [("{}/{:04}_sat.jpg".format(path,i),
                      "{}/pred/{:04}_mask.png".format(path,i)) 
                    for i in range(257)]
        self.transform = transform
        self.out_transform = out_transform 
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx][0]).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img)
        
        pred_path = self.data[idx][1]
        return img, pred_path 

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    ds = p2_Dataset('../hw2_data/p2_data/train/', transform)
    
    a,b = ds[3]
    print(b)
    
    '''
    ## count not balance
    count_ls = [0 for i in range(8)]
    for i in range(len(ds)):
        a,b = ds.__getitem__(i)
        for c in range(8):
            count_ls[c] += (b==c).sum()
    print(count_ls)
    
    for c in range(8):
        print(count_ls[2]/(count_ls[c]+1e-8))
    '''
