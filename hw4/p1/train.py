from dataset import get_transform, MiniImageNet_Dataset, CategoriesSampler, DataLoader
from model import Conv4, MLP
from utils import worker_init_fn, accuracy_, distance_metric, Averager

import torch
import torch.nn.functional as F
import os, argparse, wandb 
import numpy as np

np.random.seed(87)
torch.manual_seed(87)
torch.cuda.manual_seed_all(87)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n_batch', type=int, default=300)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train_way', type=int, default=30)
    parser.add_argument('--valid_way', type=int, default=5)
    parser.add_argument('--distance', type=str, default='parametric',
                        choices=['euclidian','cosine','parametric'])
    parser.add_argument('--device', type=str, default='cuda')
    args =  parser.parse_args() ###############################
    
    args.name = f'shot{args.shot}_trainway{args.train_way}_validway{args.valid_way}'+\
                 f'_{args.distance}_epochs{args.epochs}'
    wandb.init(config=args, project='dlcv_proto_net', name=args.name)

    # Image transform
    train_trans, valid_trans = get_transform()

    # Train data
    train_set = MiniImageNet_Dataset('../hw4_data/train/', train_trans)
    train_sampler = CategoriesSampler(train_set.label, n_batch=args.n_batch, 
                                      n_ways=args.train_way,
                                      n_shot=args.shot+args.query)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, 
                    num_workers=6, worker_init_fn=worker_init_fn)
    
    # Valid data
    valid_set = MiniImageNet_Dataset('../hw4_data/val/', valid_trans)
    valid_sampler = CategoriesSampler(valid_set.label, n_batch=args.n_batch, 
                                      n_ways=args.valid_way,
                                      n_shot=args.shot+args.query)
    valid_loader = DataLoader(valid_set, batch_sampler=valid_sampler, 
                    num_workers=6, worker_init_fn=worker_init_fn)
    
    # model
    model = Conv4().to(args.device)
    model.train()
    
    # distance F
    distance = distance_metric(args.distance, model)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_acc = 0
    train_loss, train_acc, valid_loss, valid_acc = [Averager() for i in range(4)]
    
    for epoch in range(1,args.epochs+1):
        
        ############## Train ##############
        
        model.train()
        
        for i, batch in enumerate(train_loader):
            data, _ = [d.to(args.device) for d in batch]
            p = args.shot * args.train_way
            data_support, data_query = data[:p], data[p:] #(shot*way) #(query*way)
            
            proto = model(data_support).reshape(args.shot,args.train_way,-1) 
            proto = proto.mean(dim=0) #(way,z)
            query = model(data_query) #(query*way,z)
                        
            logits = distance(query, proto) #(query*way, way)
            label = torch.arange(args.train_way).repeat(args.query).long() #(query*way)
            label = label.to(args.device)
            
            loss = F.cross_entropy(logits, label)
            acc = accuracy_(logits, label)
                        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                        
            train_loss.add(loss.item()), train_acc.add(acc)
            wandb.log({'train_loss':loss.item(), 'train_acc':acc})
            print(' [%d/%d], loss=%.3f acc=%.3f'%
                  (i+1, len(train_loader), train_loss.item(), train_acc.item()),
                      end='  \r')
        print('Epoch %d, train loss=%.4f acc=%.3f'%
              (epoch, train_loss.item(), train_acc.item()))
        
        ############## Valid ##############
        
        model.eval()
        
        for i, batch in enumerate(valid_loader):
            data, _ = [d.cuda() for d in batch]
            p = args.shot * args.valid_way
            data_support, data_query = data[:p], data[p:] #(shot*way) #(query*way)
            
            proto = model(data_support).reshape(args.shot,args.valid_way,-1)
            proto = proto.mean(dim=0) #(way,z)
            query = model(data_query) #(query*way,z)
                        
            logits = distance(query, proto) #(query*way, way)
            label = torch.arange(args.valid_way).repeat(args.query).long() #(query*way)
            label = label.to(args.device)
            
            loss = F.cross_entropy(logits, label)
            acc = accuracy_(logits, label)
                                                
            valid_loss.add(loss.item()), valid_acc.add(acc)
            print(' [%d/%d], loss=%.3f acc=%.3f'%
                  (i+1, len(valid_loader), valid_loss.item(), valid_acc.item()),
                      end='  \r')
        wandb.log({'valid_loss':valid_loss.item(), 'valid_acc':valid_acc.item()})
        print('Epoch %d, valid loss=%.4f acc=%.3f'%
                (epoch, valid_loss.item(), valid_acc.item()))
        
        if valid_acc.item() > best_acc and epoch > args.epochs//2:
            os.system('mkdir -p checkpoints')
            torch.save(model.state_dict(),f'checkpoints/{args.name}_best.pth')
            print(' save weight')
            best_acc = valid_acc.item()
        train_loss.clean(), train_acc.clean(), valid_loss.clean(), valid_acc.clean()
    
    log = f"{args.name} best_valid_acc:{best_acc}"
    with open('log.txt','a') as f:
        f.write(log+'\n')
    print(log)

