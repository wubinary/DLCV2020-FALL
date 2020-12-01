from dataset import transforms, Digits_Dataset, DataLoader
from models.dann import init_weights, FeatureExtractor, LabelPredictor , DomainClassifier 
from utils import mean_, accuracy_, circle_iterator, WarmupScheduler

import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import argparse
import wandb
import numpy as np

torch.manual_seed(987)
torch.cuda.manual_seed_all(987)
np.random.seed(987)

def train(args, dataloaders):

    train_source_dataloader = dataloaders['train_source']
    train_target_dataloader = dataloaders['train_target']

    # model
    feature_extractor = FeatureExtractor().to(args.device)
    label_predictor = LabelPredictor().to(args.device)
    models = {'F':feature_extractor,
              'C':label_predictor}
    
    # loss
    class_criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(list(label_predictor.parameters())+
                          list(feature_extractor.parameters()), lr=args.lr)

    # lr schedule 
    
    steplr_after = StepLR(optimizer, step_size=10, gamma=1.0)
    lr_scheduler = WarmupScheduler(optimizer, multiplier=1, 
                                   total_epoch=args.warmup_epochs, 
                                   after_scheduler=steplr_after)
    
    #############################################################
    ########################### Train ###########################

    train_source_dataloader_iter = circle_iterator(train_source_dataloader)
    train_target_dataloader_iter = circle_iterator(train_target_dataloader)

    best_tar_acc = 0

    for epoch in range(1, args.epochs+1):

        total_losses = []
        class_losses = []
        source_accs = []
           
        len_dataloader = min(len(train_source_dataloader),
                             len(train_target_dataloader))
        
        for i in range(len_dataloader):

            src_imgs, src_lbls = next(train_source_dataloader_iter)
            tar_imgs, tar_lbls = next(train_target_dataloader_iter)
            
            src_imgs, src_lbls = src_imgs.to(args.device),\
                                 src_lbls.to(args.device)
 
            tar_imgs, tar_lbls = tar_imgs.to(args.device),\
                                 tar_lbls.to(args.device)
            
            ######## Train Feature_extractor & Label_predictor #########
           
            features = feature_extractor(src_imgs)
            class_logits = label_predictor(features) #src
            
            C_loss = class_criterion(class_logits, src_lbls)

            loss = C_loss  
            loss.backward()

            nn.utils.clip_grad_norm_(feature_extractor.parameters(), 0.001)
            optimizer.step()
            
            optimizer.zero_grad()

            ###################      logging       #####################
            
            src_acc = accuracy_(class_logits, src_lbls)

            class_losses.append(C_loss.item())
            total_losses.append(loss.item())
            source_accs.append(src_acc)
            wandb.log({'class_loss': C_loss.item(),
                       'total_loss': loss.item(),
                       'source_acc': src_acc})
            
            if i%5 == 0:
                print(" [%d/%d] Loss Total:%.2f C:%.2f, src_acc:%.2f"
                            % ( i+1, len_dataloader,
                                loss.item(),
                                C_loss.item(),
                                src_acc), end='     \r')
        
        # lr scheduler update
        lr_scheduler.step()
       
        # validating
        valid_src_acc, valid_tar_acc = valid(args, dataloaders, models)
        wandb.log({'valid_src_acc': valid_src_acc,
                   'valid_tar_acc': valid_tar_acc})
        print(f" Epoch %d, Loss Total:%.3f C:%.3f"
               ", src_acc:%.3f tar_acc:%.3f " % (
                                epoch,
                                mean_(total_losses),
                                mean_(class_losses),
                                valid_src_acc,
                                valid_tar_acc))
        
        if valid_tar_acc > 0.2 and best_tar_acc < valid_tar_acc:
            best_tar_acc = valid_tar_acc  
            save_dir = f"./result/3_1/{args.source}2{args.target}/"
            os.system(f"mkdir -p {save_dir}")
            torch.save({'F':feature_extractor.state_dict(),
                        'C':label_predictor.state_dict()},
                    f"{save_dir}/best_model.pth")
            print("\t save best weight")


@torch.no_grad()
def valid(args, dataloaders, models=None):

    valid_source_dataloader = dataloaders['valid_source']
    valid_target_dataloader = dataloaders['valid_target']
 
    if models is None:
        load = torch.load(
                f"./result/3_1/{args.source}2{args.target}/best_model.pth")
        
        feature_extractor = FeatureExtractor()
        feature_extractor.load_state_dict(load['F'])
        feature_extractor.to(args.device)

        label_predictor = LabelPredictor()
        label_predictor.load_state_dict(load['C'])
        label_predictor.to(args.device)
    else:
        feature_extractor = models['F']
        label_predictor = models['C']
    feature_extractor.eval()
    label_predictor.eval()

    src_accs = []
    for i,(imgs, lbls) in enumerate(valid_source_dataloader):
        bsize = imgs.size(0)

        imgs, lbls = imgs.to(args.device), lbls.to(args.device)

        features = feature_extractor(imgs)
        class_output = label_predictor(features)

        acc = accuracy_(class_output, lbls)
        
        src_accs.append(acc)

        print(f"\t [{i+1}/{len(valid_target_dataloader)}]", end="  \r")


    tar_accs = []
    for i,(imgs, lbls) in enumerate(valid_target_dataloader):
        bsize = imgs.size(0)

        imgs, lbls = imgs.to(args.device), lbls.to(args.device)

        features = feature_extractor(imgs)
        class_output = label_predictor(features)

        acc = accuracy_(class_output, lbls)
        
        tar_accs.append(acc)

        print(f"\t [{i+1}/{len(valid_target_dataloader)}]", end="  \r")

    mean_src_acc, mean_tar_acc = mean_(src_accs), mean_(tar_accs)
    #print(f"\t Valid, src acc:%.3f, tar acc:%.3f" % (mean_src_acc,
    #                                                 mean_tar_acc))

    return mean_src_acc, mean_tar_acc

def parse_args(string=None):
    parser = argparse.ArgumentParser(description='Transfer Digits Dataset')
    # dataset & dataloader
    parser.add_argument('--source', type=str, default='mnistm',
                    choices=['mnistm','svhn','usps'],
                    help='source domain data')
    parser.add_argument('--target', type=str, default='svhn',
                    choices=['mnistm','svhn','usps'],
                    help='target domain data')
    parser.add_argument('--bsize', type=int, default=128,
                    help='batchsize 128')
    parser.add_argument('--num_workers', type=int, default=6,
                    help='num workers 6')
    parser.add_argument('--img_size', type=int, default=64,
                    help='trainnig image size')
    # model
    parser.add_argument('--nc', type=int, default=3,
                    help='image channels')
    parser.add_argument('--load_epoch', type=int, default=-1,
                    help='validating load trained model at epoch')
    # loss 
    parser.add_argument('--lamb', type=float, default=1.0,
                    help='loss = C_loss - lamb*D_loss')
    # optimize
    parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
    # train
    parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='epochs to warmup lr')
    parser.add_argument('--epochs', type=int, default=50,
                    help='epochs to train')
    # others
    parser.add_argument('--device', type=str, default='cuda:0',
                    help='cpu or cuda:0 or cuda:1')

    args = parser.parse_args() if string is None else parser.parse_args(string)
    return args 
    
if __name__=='__main__':
    
    args = parse_args()
   
    wandb.init(config=args, 
        project=f'dlcv_naive_{args.source}2{args.target}')

    size = 64
    t0 = transforms.Compose([
            transforms.Resize(size),
            transforms.ColorJitter(),
            transforms.RandomRotation(15, fill=(0,)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    t1 = transforms.Compose([
            transforms.Resize(size),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    root = '../hw3_data/digits/'
    # dataset
    source, target = args.source, args.target 
    train_source_dataset = Digits_Dataset(root+f'{source}/train', source, t0)
    train_target_dataset = Digits_Dataset(root+f'{target}/train', target, t0)
    valid_source_dataset = Digits_Dataset(root+f'{source}/test', source, t1)
    valid_target_dataset = Digits_Dataset(root+f'{target}/test', target, t1)
    
    # dataloaders
    train_source_dataloader = DataLoader(train_source_dataset,
                                         batch_size=args.bsize,
                                         num_workers=args.num_workers,
                                collate_fn=train_source_dataset.collate_fn,
                                         shuffle=True)
    train_target_dataloader = DataLoader(train_target_dataset,
                                         batch_size=args.bsize,
                                         num_workers=args.num_workers,
                                collate_fn=train_target_dataset.collate_fn,
                                         shuffle=True)
    valid_source_dataloader = DataLoader(valid_source_dataset,
                                         batch_size=args.bsize*4,
                                         num_workers=args.num_workers)
    valid_target_dataloader = DataLoader(valid_target_dataset,
                                         batch_size=args.bsize*4,
                                         num_workers=args.num_workers)

    dataloaders = {'train_source':train_source_dataloader,
                   'train_target':train_target_dataloader,
                   'valid_source':valid_source_dataloader,
                   'valid_target':valid_target_dataloader}

    train(args, dataloaders)
    print("\nFinish Training\n")

    src_acc, tar_acc = valid(args, dataloaders)
    print(f"Valid src_acc:%.3f tar_acc:%.3f" % (src_acc, tar_acc))

