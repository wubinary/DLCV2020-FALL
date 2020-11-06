import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from unet.unet_model import UNet
from model import VGG16_FCN32s, VGG16_FCN8s

from utils import *
from dataset import *
from mean_iou_evaluate import mean_iou_score

def _run_train(args, dataloader, model, criterion, opt):
    model.train()
    
    step_loss, step_acc, step_iou = [],[],[]
    for idx, (images, labels) in enumerate(dataloader):
        b = images.size(0)
        
        images,labels = images.to(args.device),labels.to(args.device)
        
        output = model(images)
        
        opt.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        opt.step()
        
        acc = accuracy(output,labels)
        iou = calcu_iou(output,labels)
        step_acc.append(acc)
        step_iou.append(iou)
        step_loss.append(loss.item())
        print('\t [{}/{}] loss:{:.3f} acc:{:.2f} iou:{:.2f}'.format(
                idx+1,
                len(dataloader),
                mean(step_loss),
                mean(step_acc),
                mean(step_iou))
            ,end='      \r')
        
    return mean(step_loss), mean(step_acc), mean(step_iou)

@torch.no_grad()
def _run_eval(args, dataloader, model, criterion):
    model.eval()
    
    step_loss, step_acc, step_iou = [], [],[]
    for idx, (images, labels) in enumerate(dataloader):
        b = images.size(0)
        
        images,labels = images.to(args.device),labels.to(args.device)
        
        output = model(images)
        
        loss = criterion(output,labels)
                
        acc = accuracy(output,labels)
        iou = calcu_iou(output,labels)
        step_acc.append(acc)
        step_iou.append(iou)
        step_loss.append(loss.item())
        print('\t [{}/{}] loss:{:.3f} acc:{:.2f} iou:{:.2f}'.format(
                idx+1,
                len(dataloader),
                mean(step_loss),
                mean(step_acc),
                mean(step_iou))
            ,end='      \r')        
        
    return mean(step_loss), mean(step_acc), mean(step_iou)

def train(args, train_dataloader, valid_dataloader):
    
    if str(args.model).lower()=='fcn32s':
        model = VGG16_FCN32s(n_classes=7)
    elif str(args.model).lower()=='fcn8s':
        model = VGG16_FCN8s(n_classes=7)
    else:
        model = UNet(n_channels=3, n_classes=7)
    #model = nn.DataParallel(model, device_ids=['cuda:0','cuda:1'])
    model.to(args.device)
    
    # loss
    # 0.79, 0.14, 1.0, 0.73, 2.74, 1.04, 132, 0 
    weight = torch.tensor([0.79, 0.14, 1.0, 0.73, 2.74, 1.04, 1.0])
    criterion = nn.CrossEntropyLoss(weight).to(args.device)

    # optim
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    if str(args.model)=='fcn32s': 
        milestones = [1,10,20,50]
    elif str(args.model)=='fcn8s':
        milestones = [1,10,20,60]
    else:
        milestones = [25,50,80]
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2) #learning rate decay

    best_iou = 0
    for epoch in range(args.epochs):
        print(f"\tEpoch {epoch}")
        
        loss, acc, iou = _run_train(args, train_dataloader, model, criterion, optimizer)
        print("\t train loss:{:.5f}, acc:{:.3f}, iou:{:.2f}".format(loss,acc,iou))
        
        loss, acc, iou = _run_eval(args, valid_dataloader, model, criterion)
        print("\t valid loss:{:.5f}, acc:{:.3f}, iou:{:.2f}".format(loss,acc,iou))
   
        if epoch in milestones:
            torch.save(model.state_dict(), f"./result/{epoch}_{args.model}.pth")
            print('\t [Info] save weights')
        if epoch>milestones[1] and iou>best_iou:
            best_iou = iou 
            torch.save(model.state_dict(), f"./result/best_{args.model}.pth")
            print('\t [Info] save weights')

def parse_args():
    parser = argparse.ArgumentParser(description='Image Segmentation')
    parser.add_argument('--model', type=str, default='fcn32s',
                    help='fcn32s or fcn8s or unet')
    parser.add_argument('--batch', type=int, default=8,
                    help='batchsize 8')
    parser.add_argument('--epochs', type=int, default=40,
                    help='epochs to train')
    parser.add_argument('--device', type=str, default='cuda:0',
                    help='cpu or cuda:0 or cuda:1')

    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    
    args = parse_args()

    size=512
    train_transform = transforms.Compose([
        transforms.RandomCrop(size),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    valid_transform = transforms.Compose([
        transforms.CenterCrop(size),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = p2_Dataset('../hw2_data/p2_data/train', train_transform)
    valid_dataset = p2_Dataset('../hw2_data/p2_data/validation', valid_transform, valid=True)
    
    train_dataloader = DataLoader(train_dataset,
                                 batch_size=args.batch,
                                 shuffle=True,
                                 num_workers=8)
    valid_dataloader = DataLoader(valid_dataset,
                                 batch_size=16,
                                 shuffle=False,
                                 num_workers=8)
    
    train(args, train_dataloader, valid_dataloader)
    
