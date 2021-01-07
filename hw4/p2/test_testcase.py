from model import Conv4_Hallu
from utils import distance_metric 

import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):   
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def predict(args, model, data_loader):
    
    distance = distance_metric(args.distance, model)

    prediction_results = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            data = data.to(args.device)

            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:] 
            query_input   = data[args.N_way * args.N_shot:,:,:,:]

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

            # support and query latent
            support_latent = model(support_input)
            query_latent = model(query_input)

            # prototype
            proto = support_latent.reshape(
                        args.N_way,args.N_shot,-1)
            proto = proto.permute(1,0,2) #(shot,way,z)
            proto_hallu = model.hallucinate(proto.mean(dim=0), 
                                                args.hallu_m)
            
            proto_aug = torch.cat((proto,proto_hallu),dim=0) #(shot+m,way,z)
            proto_aug = proto_aug.mean(dim=0) #(way,z)
            #proto_aug = proto.mean(dim=0)

            # distances
            logits = distance(query_latent, proto_aug)
            
            _, indices = torch.topk(logits, k=1, dim=1)
            indices = indices.view(-1).cpu().numpy()
            
            prediction_results.append(indices)

            print(f"[{i}/{len(data_loader)}]", end="  \r")

    prediction_results = np.array(prediction_results)
    index = np.array([[i] for i in range(len(prediction_results))])
    prediction_results = np.concatenate(
                            (index,prediction_results), axis=1)
    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--hallu_m', default=20, type=int, help='hallu m data')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file",
                        default='../hw4_data/val.csv')
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory",
                        default='../hw4_data/val/')
    parser.add_argument('--testcase_csv', type=str, help="Test case csv",
                        default='../hw4_data/val_testcase.csv')
    parser.add_argument('--output_csv', type=str, help="Output filename",
                        default='./pred.csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--distance', type=str, default='parametric',
                        choices=['euclidian','cosine','parametric'])
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    model = Conv4_Hallu()
    model.load_state_dict(torch.load(args.load, map_location='cpu'))
    model.to(args.device)
    model.eval()

    prediction_results = predict(args, model, test_loader)

    N_query = len(prediction_results[0])-1
    df_pred = pd.DataFrame(prediction_results, columns=['episode_id']+\
                                            [f'query{i}' for i in range(N_query)])
    df_pred.to_csv(args.output_csv, index=False)

    # acc 
    #df_gt = pd.read_csv(str(args.testcase_csv).replace('.csv','_gt.csv'))
    #total_len = df_gt.to_numpy().shape[0] * df_gt.to_numpy().shape[1]
    #test_acc = (df_gt.to_numpy()==df_pred.to_numpy()).sum() / total_len
    #print(f"test_acc:{test_acc}")
    
