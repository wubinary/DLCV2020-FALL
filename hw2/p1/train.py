import torch
import torch.nn as nn
import torch.optim as optim
import argparse 

import torchvision.models as md 
from models.mobilenetv2 import MobileNetV2
from models.resnet import resnet18
from models.vgg import vgg11_bn, vgg7_bn

from utils import *
from dataset import *
from model import Model

def _run_train(args, dataloader, model, criterion, opt):
    model.train()
    
    step_loss, step_acc = [],[]
    for idx, (images, labels) in enumerate(dataloader):
        b = images.size(0)
        
        images,labels = images.to(args.device),labels.to(args.device)
        
        output = model(images)
        
        opt.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        opt.step()
        
        acc = accuracy(output,labels)
        step_acc.append(acc)
        step_loss.append(loss.item())
        print('\t [{}/{}] loss:{:.3f} acc:{:.2f}'.format(
                idx+1,
                len(dataloader),
                mean(step_loss),
                mean(step_acc))
            ,end='      \r')
        
    return mean(step_loss), mean(step_acc)

@torch.no_grad()
def _run_eval(args, dataloader, model, criterion):
    model.eval()
    
    step_loss, step_acc = [],[]
    for idx, (images, labels) in enumerate(dataloader):
        b = images.size(0)
        
        images,labels = images.to(args.device),labels.to(args.device)
        
        output = model(images)
        
        loss = criterion(output, labels)
                
        acc = accuracy(output,labels)
        step_acc.append(acc)
        step_loss.append(loss.item())
        print('\t [{}/{}] loss:{:.3f} acc:{:.2f}'.format(
                idx+1,
                len(dataloader),
                mean(step_loss),
                mean(step_acc))
            ,end='      \r')        
        
    return mean(step_loss), mean(step_acc)

def train(args, train_dataloader, valid_dataloader):
    
    model = Model(50)
    #model = nn.DataParallel(model, device_ids=['cuda:0','cuda:1'])
    model.to(args.device)
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.01)
    
    milestones = [5,10]
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2) #learning rate decay

    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\tEpoch {epoch}")
        
        loss, acc = _run_train(args, train_dataloader, model, criterion, optimizer)
        print("\t train loss:{:.5f}, acc:{:.3f}".format(loss,acc))
        
        loss, acc = _run_eval(args, valid_dataloader, model, criterion)
        print("\t valid loss:{:.5f}, acc:{:.3f}".format(loss,acc))
    
        if epoch>milestones[0] and acc>best_acc:
            best_acc = acc 
            torch.save(model.state_dict(), "./result/best.pt")
            print('\t [Info] save weights')

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--epochs', type=int, default=60,
                    help='epochs to train')
    parser.add_argument('--device', type=str, default='cuda:0',
                    help='cpu or cuda:0 or cuda:1')

    args = parser.parse_args()
    return args 

if __name__ == '__main__':

    args = parse_args()
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=128),#, scale=(0.5, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = p1_Dataset('../hw2_data/p1_data/train_50/', train_transform)
    valid_dataset = p1_Dataset('../hw2_data/p1_data/val_50/', valid_transform)
    
    train_dataloader = DataLoader(train_dataset,
                                 batch_size=128,
                                 shuffle=True,
                                 num_workers=8)
    valid_dataloader = DataLoader(valid_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 num_workers=8)
    
    train(args, train_dataloader, valid_dataloader)
    
