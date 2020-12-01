from dataset import transforms, Digits_Dataset, DataLoader
from models.model import init_weights, Model
#FeatureExtractor, LabelPredictor
from models.DomainClassifierSource import DClassifierForSource
from models.DomainClassifierTarget import DClassifierForTarget
from models.EntropyMinimizationPrinciple import EMLossForTarget
from models.resnet import resnet 
from utils import mean_, accuracy_, circle_iterator, WarmupScheduler

import os
import math
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
    model = resnet(args).to(args.device)
    #model = Model().to(args.device)
    model.train()

    # loss
    Loss_Src_Class = DClassifierForTarget(
                            nClass=args.num_classes).to(args.device)
    Loss_Tar_Class = DClassifierForSource(
                            nClass=args.num_classes).to(args.device)
    Loss_Tar_EM = EMLossForTarget(
                            nClass=args.num_classes).to(args.device)
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()
    
    models = model 
    
    # optimizer
    
    optimizer = torch.optim.SGD([
     {'params': model.conv1.parameters(), 'name': 'pre-trained'},
     {'params': model.bn1.parameters(), 'name': 'pre-trained'},
     {'params': model.layer1.parameters(), 'name': 'pre-trained'},
     {'params': model.layer2.parameters(), 'name': 'pre-trained'},
     {'params': model.layer3.parameters(), 'name': 'pre-trained'},
     {'params': model.layer4.parameters(), 'name': 'pre-trained'},
     #{'params': model.fc.parameters(), 'name': 'pre-trained'}
     {'params': model.fc.parameters(), 'name': 'new-added'}
        ],
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay, 
                nesterov=True)  
    '''
    optimizer = torch.optim.SGD(model.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay),
                    nesterov=True)
    '''
    # lr schedule 
    '''
    steplr_after = StepLR(optimizer, step_size=10, gamma=1.0)
    lr_scheduler = WarmupScheduler(optimizer, multiplier=1, 
                                   total_epoch=args.warmup_epochs, 
                                   after_scheduler=steplr_after)
    '''
    #############################################################
    ########################### Train ###########################

    train_source_dataloader_iter = circle_iterator(train_source_dataloader)
    train_target_dataloader_iter = circle_iterator(train_target_dataloader)

    best_tar_acc = 0

    for epoch in range(1, args.epochs+1):

        class_losses = []
        generate_losses = []
                
        len_dataloader = min(len(train_source_dataloader),
                             len(train_target_dataloader))
        
        model.train()
        
        for i in range(len_dataloader):

            src_imgs, src_lbls = next(train_source_dataloader_iter)
            tar_imgs, tar_lbls = next(train_target_dataloader_iter)
        
        #for i,((src_imgs, src_lbls), (tar_imgs, tar_lbls)) in enumerate(
        #        zip(train_source_dataloader, train_target_dataloader)):
           
            src_imgs, src_lbls = src_imgs.to(args.device),\
                                 src_lbls.to(args.device)
 
            tar_imgs, tar_lbls = tar_imgs.to(args.device),\
                                tar_lbls.to(args.device) 
            
            ################### Train Domain_predictor ###################
            
            src_lbls_tmp = src_lbls + args.num_classes 

            output = model(src_imgs)
            loss_Cs_src = CE(output[:,:args.num_classes], src_lbls)
            loss_Ct_src = CE(output[:,args.num_classes:], src_lbls)

            loss_Cst_domain_part1 = Loss_Src_Class(output)
            loss_Cst_category_Gen = 0.5 * CE(output, src_lbls) + \
                                    0.5 * CE(output, src_lbls_tmp)

            output = model(tar_imgs)
            loss_Cst_domain_part2 = Loss_Tar_Class(output)
            loss_Cst_domain_Gen = 0.5 * Loss_Tar_Class(output) + \
                                  0.5 * Loss_Src_Class(output)
            loss_em_tar = Loss_Tar_EM(output)

            ########################### Loss ############################
            lamb = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1
            #lamb = 0.1
            
            if args.flag == 'no_em':
                C_loss = loss_Cs_src + loss_Ct_src + \
                            loss_Cst_domain_part1 + loss_Cst_domain_part2
                G_loss = loss_Cst_category_Gen + \
                            lamb * (loss_Cst_domain_Gen)
            elif args.flag == 'symnet':
                lamb = 2 / (1 + math.exp(-1 * 1 * epoch / args.epochs)) - 1
                #lamb = 2 / (1 + math.exp(-1 * 0.1 * epoch / args.epochs)) - 1
                C_loss = loss_Cs_src + loss_Ct_src + \
                            loss_Cst_domain_part1 + loss_Cst_domain_part2
                G_loss = loss_Cst_category_Gen + \
                            lamb * (loss_Cst_domain_Gen + loss_em_tar)
            elif args.flag == 'test':
                lamb = 2 / (1 + math.exp(-1 * 5 * epoch / args.epochs)) - 1
                C_loss = (loss_Cs_src + loss_Ct_src) + \
                            loss_Cst_domain_part1 + loss_Cst_domain_part2
                G_loss = loss_Cst_category_Gen + \
                            lamb * (loss_Cst_domain_Gen + loss_em_tar)
            else:
                 raise ValueError('unrecognized flag:', args.flag)
            '''  
            #nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.zero_grad()
            C_loss.backward(retain_graph=True)
            optimizer.step()

            optimizer.zero_grad()
            G_loss.backward()
            optimizer.step()
            
            '''
            #compute gradient and do SGD step
            optimizer.zero_grad()
            C_loss.backward(retain_graph=True)
            temp_grad = []
            for param in model.parameters():
                temp_grad.append(param.grad.data.clone())
            grad_for_classifier = temp_grad

            optimizer.zero_grad()
            G_loss.backward()
            temp_grad = []
            for param in model.parameters():
                 temp_grad.append(param.grad.data.clone())
            grad_for_featureExtractor = temp_grad

            #print(list(model.parameters())[-2:],len(list(model.parameters())))
            #input()
            count = 0
            for param in model.parameters():
                temp_grad = param.grad.data.clone()
                temp_grad.zero_()
                if count < 60: #159 #### feauter extrac of the ResNet-50
                    #60 #### Resnet18
                    #5層: 20
                    temp_grad = temp_grad + grad_for_featureExtractor[count]
                else:
                    temp_grad = temp_grad + grad_for_classifier[count]
                temp_grad = temp_grad
                param.grad.data = temp_grad
                count = count + 1
            optimizer.step()
             
            ###################      logging       #####################
            
            class_losses.append(C_loss.item())
            generate_losses.append(G_loss.item())
            wandb.log({'class_loss': C_loss.item(),
                       'generate_loss': G_loss.item()})
            
            if i%5 == 0:
                print(" [%d/%d] Loss C:%.2f G:%.2f"
                            % ( i+1, len_dataloader,
                                C_loss.item(),
                                G_loss.item()), end='     \r')
        
        # lr scheduler update
        #lr_scheduler.step()
       
        # validating
        valid_src_acc, valid_tar_acc, valid_tar_acc_mix = valid(args, dataloaders, models)
        wandb.log({'valid_src_acc': valid_src_acc,
                   'valid_tar_acc': valid_tar_acc,
                   'valid_tar_acc_mix': valid_tar_acc_mix})
        print(f" Epoch %d, Loss C:%.3f G:%.3f"
               ", src_acc:%.3f tar_acc1:%.3f tar_acc2:%.3f " % (
                                epoch,
                                mean_(class_losses),
                                mean_(generate_losses),
                                valid_src_acc,
                                valid_tar_acc,
                                valid_tar_acc_mix))
        
        if valid_tar_acc > 0.2 and best_tar_acc < valid_tar_acc:
            best_tar_acc = valid_tar_acc  
            save_dir = f"./result/{args.source}2{args.target}/"
            os.system(f"mkdir -p {save_dir}")
            torch.save({'M':model.state_dict()},
                    f"{save_dir}/best_model.pth")
            print("\t save best weight")


@torch.no_grad()
def valid(args, dataloaders, models=None):

    valid_source_dataloader = dataloaders['valid_source']
    valid_target_dataloader = dataloaders['valid_target']
 
    if models is None:
        try:
            load = torch.load(
                f"./result/{args.source}2{args.target}/best_model.pth")
        
            model = resnet(args)
            #model = Model()
            model.load_state_dict(load['M'])
        except Exception as inst:
            model = resnet(args)
        model.eval()
        model.to(args.device)

    else:
        model = models 
    model.eval()

    src_accs = []
    for i,(imgs, lbls) in enumerate(valid_source_dataloader):
        bsize = imgs.size(0)

        imgs, lbls = imgs.to(args.device), lbls.to(args.device)

        output = model(imgs)

        acc = accuracy_(output[:,:args.num_classes], lbls)
        
        src_accs.append(acc)

        print(f"\t [{i+1}/{len(valid_target_dataloader)}]", end="  \r")


    tar_accs1 = []
    tar_accs2 = []
    for i,(imgs, lbls) in enumerate(valid_target_dataloader):
        bsize = imgs.size(0)

        imgs, lbls = imgs.to(args.device), lbls.to(args.device)

        output = model(imgs)

        acc = accuracy_(output[:,args.num_classes:], lbls)        
        tar_accs1.append(acc)

        acc = accuracy_(output[:,args.num_classes:]+
                        output[:,:args.num_classes], lbls)        
        tar_accs2.append(acc)

        print(f"\t [{i+1}/{len(valid_target_dataloader)}]", end="  \r")

    mean_src_acc, mean_tar_acc1, mean_tar_acc2 = mean_(src_accs), mean_(tar_accs1), mean_(tar_accs2)
    #print(f"\t Valid, src acc:%.3f, tar acc:%.3f" % (mean_src_acc,
    #                                                 mean_tar_acc))

    return mean_src_acc, mean_tar_acc1, mean_tar_acc2

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
    parser.add_argument('--img_size', type=int, default=32,
                    help='trainnig image size')
    parser.add_argument('--num_classes', type=int, default=10,
                    help='num classes')
    # model
    parser.add_argument('--nc', type=int, default=3,
                    help='image channels')
    parser.add_argument('--load_epoch', type=int, default=-1,
                    help='validating load trained model at epoch')
    parser.add_argument('--arch', type=str, default='resnet18', 
                    help='Model name')
    parser.add_argument('--pretrained', action='store_true', 
                    help='whether using pretrained model')
    # loss 
    parser.add_argument('--lamb', type=float, default=1.0,
                    help='loss = C_loss - lamb*D_loss')
    parser.add_argument('--flag', type=str, default='symnet',
                    #choices=['symnet','no_em'], 
                        help='loss mode')
    # optimize
    parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, 
                    help='Momentum.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, 
                    help='Weight decay (L2 penalty).')
    # train
    parser.add_argument('--warmup_epochs', type=int, default=10,
                    help='epochs to warmup lr')
    parser.add_argument('--epochs', type=int, default=100,
                    help='epochs to train')
    # others
    parser.add_argument('--device', type=str, default='cuda:0',
                    help='cpu or cuda:0 or cuda:1')

    args = parser.parse_args() if string is None else parser.parse_args(string)
    return args 
    
if __name__=='__main__':
    
    args = parse_args()
   
    wandb.init(config=args, 
            project=f'dlcv_symnet_{args.source}2{args.target}')

    size = args.img_size
    t0 = transforms.Compose([
            transforms.Resize(size),
            transforms.ColorJitter(),
            #transforms.RandomRotation(15, fill=(0,)),
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

    src_acc, tar_acc1, tar_acc2 = valid(args, dataloaders)
    print(f"Valid src_acc:%.3f tar_acc:%.3f %3f" % (src_acc, 
                                                    tar_acc1,tar_acc2))

