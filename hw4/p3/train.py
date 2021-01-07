from dataset import get_transform, MiniImageNet_Dataset, CategoriesSampler, DataLoader
from model import Conv4_DTN
from utils import worker_init_fn, accuracy_, distance_metric, Averager
from utils import WarmupScheduler 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import os, argparse, wandb 
import numpy as np

np.random.seed(87)
torch.manual_seed(87)
torch.cuda.manual_seed_all(87)

def train_base(args, base_loader, model, criterion, optimizer, epoch):

    losses, accs = [Averager() for i in range(2)]

    model.train()
    for i, (imgs, lbls) in enumerate(base_loader):
        imgs,lbls = imgs.to(args.device),lbls.to(args.device)

        features = model(imgs)
        predicts = model.classify(features)

        loss = criterion(predicts, lbls)
        acc = accuracy_(predicts, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.weight_norm()
        
        losses.add(loss.item()), accs.add(acc)
        print(" [%d/%d] base loss:%.3f, acc:%.2f" % 
                (i+1,len(base_loader),losses.item(),accs.item()), end='  \r')
    
    wandb.log({'base_train_loss':losses.item(),'base_train_acc':accs.item()})
    print("Epoch %d, base loss:%.4f, acc:%.3f" % 
            (epoch,losses.item(),accs.item()))

def train(args, train_loader, model, distance, optimizer, epoch):
    model.train()
        
    for i, batch in enumerate(train_loader):
        data, _ = [d.to(args.device) for d in batch]
        p = args.shot * args.train_way
        data_support, data_query = data[:p], data[p:] #(shot*way) #(query*way)
            
        # proto
        proto = model(data_support)
        proto = proto.reshape(args.shot,args.train_way,-1)
            
        proto_hallu = model.hallucinate(proto.mean(dim=0), args.hallu_m)

        proto_aug = torch.cat((proto,proto_hallu),dim=0) #(shot+m,way,z)
        proto_aug = proto_aug.mean(dim=0) #(way,z)

        # query
        query = model(data_query) #(query*way,z)
        query = query.view(1,args.query*args.train_way,-1)

        query_hallu = model.hallucinate(query.squeeze(0), args.hallu_m)

        query_aug = torch.cat((query,query_hallu),dim=0) #((1+m),query*way,z)
        query_aug = query_aug.view((1+args.hallu_m)*args.query*args.train_way, -1)

        # distance
        logits = distance(query_aug, proto_aug) #((1+m)*query*way, way)
        label = torch.arange(args.train_way).repeat(
                            (1+args.hallu_m)*args.query).long() #(query*way)
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
    print('Epoch %d, Meta-train loss=%.4f acc=%.3f'%
          (epoch, train_loss.item(), train_acc.item()))
    train_loss.clean(),train_acc.clean()

@torch.no_grad()
def valid(args, valid_loader, model, distance, epoch):
    
    model.eval()

    if not hasattr(valid, "best_acc"): 
        valid.best_acc = 0

    for i, batch in enumerate(valid_loader):
        data, _ = [d.cuda() for d in batch]
        p = args.shot * args.valid_way
        data_support, data_query = data[:p], data[p:] #(shot*way) #(query*way)
        
        # proto
        proto = model(data_support).reshape(args.shot,args.valid_way,-1)
        proto = proto.reshape(args.shot,args.valid_way,-1)

        proto_hallu = model.hallucinate(proto.mean(dim=0), args.hallu_m)

        proto_aug = torch.cat((proto,proto_hallu),dim=0) #(shot+m,way,z)
        proto_aug = proto_aug.mean(dim=0) #(way,z)
 
        # query
        query = model(data_query) #(query*way,z)
        query = query.view(1,args.query*args.valid_way,-1)

        #query_hallu = model.hallucinate(query.squeeze(0), args.hallu_m)

        #query_aug = torch.cat((query,query_hallu),dim=0) #((1+m),query*way,z)
        #query_aug = query_aug.view((1+args.hallu_m)*args.query*args.valid_way, -1)
        query_aug = query.squeeze(0)

        # distance
        logits = distance(query_aug, proto_aug) #(query*way, way)
        label = torch.arange(args.valid_way).repeat(
                        #(1+args.hallu_m)*
                        args.query).long() #(query*way)
        label = label.to(args.device)
            
        loss = F.cross_entropy(logits, label)
        acc = accuracy_(logits, label)
                                                
        valid_loss.add(loss.item()), valid_acc.add(acc)
        print(' [%d/%d], loss=%.3f acc=%.3f'%
              (i+1, len(valid_loader), valid_loss.item(), valid_acc.item()),
                  end='  \r')
    wandb.log({'valid_loss':valid_loss.item(), 'valid_acc':valid_acc.item()})
    print('Epoch %d, Meta-valid loss=%.4f acc=%.3f'%
            (epoch, valid_loss.item(), valid_acc.item()))
       
    if valid_acc.item() > valid.best_acc and epoch > args.epochs//2:
        os.system('mkdir -p checkpoints')
        torch.save(model.state_dict(),f'checkpoints/{args.name}_best.pth')
        print(' save weight')
        valid.best_acc = valid_acc.item()
    
    valid_loss.clean(), valid_acc.clean()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--n_batch', type=int, default=300)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train_way', type=int, default=30)
    parser.add_argument('--valid_way', type=int, default=5)
    parser.add_argument('--hallu_m', type=int, default=20)
    parser.add_argument('--distance', type=str, default='parametric',
                        choices=['euclidian','cosine','parametric'])
    parser.add_argument('--device', type=str, default='cuda')
    args =  parser.parse_args() ###############################
    
    args.name = f'DTN_hallu{args.hallu_m}_shot{args.shot}_trainway{args.train_way}'+\
                f'_validway{args.valid_way}_{args.distance}'
    wandb.init(config=args, project='dlcv_proto_net', name=args.name)

    # Image transform
    train_trans, valid_trans = get_transform()
    
    # Base Train data
    base_train_set = MiniImageNet_Dataset('../hw4_data/train/', train_trans)
    base_train_loader = DataLoader(base_train_set, batch_size=64, 
                                    num_workers=6, shuffle=True)

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
    model = Conv4_DTN().to(args.device)
    model.train()
    
    # distance F
    distance = distance_metric(args.distance, model)

    # criterion classify
    ce_loss = nn.CrossEntropyLoss()
    
    # optimizer
    optimizer1 = torch.optim.Adam(model.parameters(), lr=2e-3)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-4)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    steplr_after = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, 
                        T_max=args.epochs, eta_min=1e-4)
    lr_scheduler = WarmupScheduler(optimizer1, multiplier=1, 
                        total_epoch=args.warmup_epochs, 
                        after_scheduler=steplr_after)
    
    best_acc = 0
    train_loss, train_acc, valid_loss, valid_acc = [Averager() for i in range(4)]
    
    for epoch in range(1,args.epochs+1):

        train_base(args, base_train_loader, model, ce_loss, optimizer2, epoch)
        
        train(args, train_loader, model, distance, optimizer1, epoch)

        valid(args, valid_loader, model, distance, epoch)
 
        lr_scheduler.step()
    
