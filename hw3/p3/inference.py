from dataset import transforms, Digits_Dataset_Test, DataLoader
from models.dann import init_weights, FeatureExtractor, LabelPredictor , DomainClassifier

import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import argparse
import numpy as np
import pandas as pd

torch.manual_seed(987)
torch.cuda.manual_seed_all(987)
np.random.seed(987)


def parse_args(string=None):
    parser = argparse.ArgumentParser()
    # testing
    parser.add_argument('--dataset_path', type=str,
                        default='hw3_data/digits/usps/test',
                        help='testing images path')
    parser.add_argument('--target', type=str, default='svhn',
                    choices=['mnistm','svhn','usps'],
                    help='target domain data')
    parser.add_argument('--out_csv', type=str,
                        default='test_output',
                        help='testing images path')
    parser.add_argument('--img_size', type=int, default=64,
                    help='trainnig image size')
    # others
    parser.add_argument('--device', type=str, default='cuda:0',
                    help='cpu or cuda:0 or cuda:1')

    args = parser.parse_args() if string is None else parser.parse_args(string)
    return args 

@torch.no_grad()
def inference(args):
    
    if args.target=='mnistm':
        args.source = 'usps'
    elif args.target=='usps':
        args.source = 'svhn'
    elif args.target=='svhn':
        args.source = 'mnistm'
    else:
        raise NotImplementedError(f"{args.target}: not implemented!")
    
    size = args.img_size
    t1 = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    valid_target_dataset = Digits_Dataset_Test(args.dataset_path, t1)
        
    valid_target_dataloader = DataLoader(valid_target_dataset,
                                             batch_size=512,
                                             num_workers=6)
        
         
    load = torch.load(
        f"./p3/result/3_2/{args.source}2{args.target}/best_model.pth",
        map_location='cpu')
        
    feature_extractor = FeatureExtractor()
    feature_extractor.load_state_dict(load['F'])
    feature_extractor.cuda()
    feature_extractor.eval()

    label_predictor = LabelPredictor()
    label_predictor.load_state_dict(load['C'])
    label_predictor.cuda()
    label_predictor.eval()
           
    out_preds = []
    out_fnames = []
    count=0
    for i,(imgs, fnames) in enumerate(valid_target_dataloader):
        bsize = imgs.size(0)

        imgs = imgs.cuda()

        features = feature_extractor(imgs)
        class_output = label_predictor(features)
        
        _, preds = class_output.max(1)
        preds = preds.detach().cpu()
        
        out_preds.append(preds)
        out_fnames += fnames
        
        count+=bsize
        print(f"\t [{count}/{len(valid_target_dataloader.dataset)}]", 
                                                        end="   \r")
        
    out_preds = torch.cat(out_preds)
    out_preds = out_preds.cpu().numpy()
    
    d = {'image_name':out_fnames, 'label':out_preds}
    df = pd.DataFrame(data=d)
    df = df.sort_values('image_name')
    df.to_csv(args.out_csv, index=False)
    print(f' [Info] finish predicting {args.dataset_path}')
    
    
    
if __name__=='__main__':
    
    args = parse_args()
    inference(args)
