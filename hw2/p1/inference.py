import os,argparse 
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from dataset import *
from model import Model

@torch.no_grad()
def inference(args, dataloader):
    
    # dataframe
    df = pd.DataFrame(data={'image_id':[],'label':[]})
    df['image_id'] = df['image_id'].astype(str)
    df['label'] = df['label'].astype(np.int)
    
    # model
    model = Model(50)
    model.load_state_dict(torch.load('./p1/result/best.pt', 
                                     map_location='cpu'))
    model.to(args.device)
    model.eval()
    
    pivote=0
    for idx, (images, path) in enumerate(dataloader):
        b = images.size(0)
        
        images = images.to(args.device)
        
        output = model(images)
        output = output.max(-1)[1].cpu().numpy().tolist()
        
        for s in range(b):
            df = df.append({'image_id':path[s].replace(dataloader.dataset.path,'').replace('/',''),
                            'label':output[s]}, 
                           ignore_index=True)
            
        pivote += b
        print(f'\t[{pivote}/{len(dataloader.dataset)}]', end='  \r')
    
    df.to_csv(os.path.join(args.out_csv,"test_pred.csv"), index=False)
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--device', type=str, default='cuda',
                    help='cpu or cuda:0 or cuda:1')
    parser.add_argument('--test_dir', type=str, default='../hw2_data/p1_data/val_50/',
                    help='cpu or cuda:0 or cuda:1')
    parser.add_argument('--out_csv', type=str, default='./test_pred.csv',
                    help='cpu or cuda:0 or cuda:1')

    args = parser.parse_args()
    return args 

if __name__ == '__main__':

    args = parse_args()
    
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = p1_Dataset_test(args.test_dir, transform)
    
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=False,
                            num_workers=8)
    
    inference(args, dataloader)
    

    
