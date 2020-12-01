from dataset import transforms, Face_Dataset, DataLoader
from model import init_weights, Generator, Discriminator
from utils import mean_

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import wandb
import os 

def train(args, train_dataloader, valid_dataloader):
    # model
    netG = Generator(args.z_dim, args.dim).to(args.device)
    netG.apply(init_weights)
    
    netD = Discriminator(args.nc, args.dim).to(args.device)
    netD.apply(init_weights)
    
    # loss
    criterion = nn.BCELoss()

    # real & fake labels 
    real_label = 1.
    fake_label = 0.

    # optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    #############################################################
    ########################### Train ###########################
    iters = 0

    for epoch in range(1,args.epochs+1):
        img_list = []
        G_losses = []
        D_losses = []
        D_real_acc_s = []
        D_fake_acc_s = []

        for i, real_imgs in enumerate(train_dataloader):
            b_size = real_imgs.size(0)
            
            ##################################
            # Part-I : Train Discriminator
            ##################################
            
            optimizerD.zero_grad()
            
            real_labels = torch.zeros(0).new_full(
                        (b_size,), real_label).to(args.device) 
            fake_labels = torch.zeros(0).new_full(
                        (b_size,), fake_label).to(args.device)
             
            ## real-batch            
            real_imgs = real_imgs.to(args.device)
           
            output = netD(real_imgs).view(-1)
            
            D_real_loss = criterion(output, real_labels)
            D_real_loss.backward()
            D_real_acc = output.mean().item()

            ## fake-batch
            noise = torch.randn(b_size, args.z_dim, device=args.device)
            fake_imgs = netG(noise) 
            output = netD(fake_imgs.detach()).view(-1)
           
            D_fake_loss = criterion(output, fake_labels)
            D_fake_loss.backward()
            D_fake_acc = output.mean().item()
            
            ## real+fake discrimator loss
            D_loss = D_real_loss + D_fake_loss
           
            optimizerD.step()
        
            ##################################
            # Part-II : Train Generator
            ##################################
            
            optimizerG.zero_grad()
            
            output = netD(fake_imgs).view(-1)
            
            G_loss = criterion(output, real_labels)
            G_loss.backward()
            
            optimizerG.step()
       
            ##################################

            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())
            D_real_acc_s.append(D_real_acc)
            D_fake_acc_s.append(D_fake_acc)
            wandb.log({'G_loss':G_loss.item(),
                       'D_loss':D_loss.item(),
                       'D_real_acc':D_real_acc,
                       'D_fake_acc':D_fake_acc})

            print(f"\t[%d/%d] LossG:%.2f, LossD:%.2f, D(x):%.2f, D(G(x)):%.2f" % (
                                i+1, len(train_dataloader),
                                G_loss.item(), D_loss.item(),
                                D_real_acc, D_fake_acc), end='     \r')
        
        print(f"\t Epoch %d, LossG:%.3f, LossD:%.3f, D(x):%.3f, D(G(x)):%.3f" % (
                                epoch,
                                mean_(G_losses), mean_(D_losses),
                                mean_(D_real_acc_s), mean_(D_fake_acc_s)))
        if epoch%10 == 0:
            os.system('mkdir -p result')
            torch.save(netD.state_dict(), f"./result/{epoch}_netD.pth")
            torch.save(netG.state_dict(), f"./result/{epoch}_netG.pth")
            print("\t save weight")

def parse_args(string=None):
    parser = argparse.ArgumentParser(description='GAN FaceDataset')
    # dataloader
    parser.add_argument('--batch', type=int, default=128,
                    help='batchsize 128')
    parser.add_argument('--num_workers', type=int, default=8,
                    help='num workers 8')
    # model
    parser.add_argument('--img_size', type=int, default=64,
                    help='image size')
    parser.add_argument('--nc', type=int, default=3,
                    help='image channels')
    parser.add_argument('--z_dim', type=int, default=100,
                    help='generator input latent')
    parser.add_argument('--dim', type=int, default=64,
                    help='generator feature maps')
    # optimize
    parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                    help='learning rate')
    parser.add_argument('--weight_cliping_limit', type=float, default=1e-2,
                    help='Wasseterin model weight clipping')
    # train
    parser.add_argument('--epochs', type=int, default=300,
                    help='epochs to train')
    # others
    parser.add_argument('--device', type=str, default='cuda:0',
                    help='cpu or cuda:0 or cuda:1')

    args = parser.parse_args() if string is None else parser.parse_args(string)
    return args 
    
if __name__=='__main__':
    
    args = parse_args()
   
    wandb.init(config=args, project='dlcv_gan_face')
    
    transform = transforms.Compose([
                    transforms.Resize(args.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3)
                ])
    train_dataset = Face_Dataset('../hw3_data/face/train', transform)
    valid_dataset = Face_Dataset('../hw3_data/face/test', transform)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch,
                                  shuffle=True,
                                  num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=args.batch,
                                  num_workers=args.num_workers)
    
    train(args, train_dataloader, valid_dataloader)
