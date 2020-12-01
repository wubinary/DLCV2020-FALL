from dataset import transforms, Face_Dataset, DataLoader
from model import init_weights, Generator, Discriminator
from utils import mean_

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import wandb

def train(args, train_dataloader, valid_dataloader):
    # model
    netG = Generator(args.nc, args.nz, args.ngf).to(args.device)
    netG.apply(init_weights)
    
    netD = Discriminator(args.nc, args.ndf).to(args.device)
    netD.apply(init_weights)
    
    # loss
    criterion = nn.BCELoss()

    # batch of latent vectors
    # input of generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=args.device)

    # real & fake labels 
    real_label = 1.
    fake_label = 0.

    # optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    #optimizerG = optim.SGD(netG.parameters(), lr=args.lr)
    #optimizerD = optim.SGD(netD.parameters(), lr=args.lr)

    #############################################################
    ########################### Train ###########################
    img_list = []
    G_losses = []
    D_losses = []
    D_real_acc_s = []
    D_fake_acc_s = []
    iters = 0

    one = torch.FloatTensor([1])
    m_one = one * -1
    one = one.to(args.device)
    m_one = m_one.to(args.device)

    for epoch in range(1,args.epochs+1):

        for i, real_imgs in enumerate(train_dataloader):
            b_size = real_imgs.size(0)
            
            ##################################
            # Part-I : Train Discriminator
            ##################################
            
            optimizerD.zero_grad()
           
            # clamp weight to range, WGAN
            for p in netD.parameters():
                p.data.clamp_(-args.weight_cliping_limit, args.weight_cliping_limit) 
            
            ## real-batch            
            real_imgs = real_imgs.to(args.device)
            label = torch.zeros(0).new_full(
                        (b_size,), real_label).to(args.device)
            
            output = netD(real_imgs).view(-1)
            
            #D_real_loss = criterion(output, label)
            #D_real_loss.backward()
            #D_real_acc = output.mean().item()
            D_real_loss = output.mean().view(1)
            D_real_loss.backward(m_one) 
            D_real_acc = (output.detach().cpu().numpy()>0).sum()/b_size

            ## fake-batch
            label = torch.zeros(0).new_full(
                        (b_size,), fake_label).to(args.device)
            
            noise = torch.randn(b_size, args.nz, 1, 1, device=args.device)
            fake_imgs = netG(noise)
            
            output = netD(fake_imgs.detach()).view(-1)
           
            #D_fake_loss = criterion(output, label)
            #D_fake_loss.backward()
            #D_fake_acc = output.mean().item()
            D_fake_loss = output.mean().view(1)
            D_fake_loss.backward(one)
            D_fake_acc = (output.detach().cpu().numpy()<0).sum()/b_size
            
            ## real+fake discrimator loss
            D_loss = D_real_loss + D_fake_loss
           
            optimizerD.step()
        
            ##################################
            # Part-II : Train Generator
            ##################################
            
            optimizerG.zero_grad()
            
            # clamp weight to range, WGAN
            for p in netG.parameters():
                p.data.clamp_(-args.weight_cliping_limit, args.weight_cliping_limit)    
        
            label = torch.zeros(0).new_full(
                        (b_size,), real_label).to(args.device)
            
            #noise = torch.randn(b_size, args.nz, 1, 1, device=args.device) 
            #fake_imgs = netG(noise)
            output = netD(fake_imgs).view(-1)
            
            #G_loss = criterion(output, label)
            #G_loss.backward()
            G_loss = output.mean().view(1)
            G_loss.backward(m_one)
            
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
    parser.add_argument('--nc', type=int, default=3,
                    help='image channels')
    parser.add_argument('--nz', type=int, default=100,
                    help='generator input latent')
    parser.add_argument('--ngf', type=int, default=64,
                    help='generator feature maps')
    parser.add_argument('--ndf', type=int, default=64,
                    help='discriminator feature maps')
    # optimize
    parser.add_argument('--lr', type=float, default=2e-4,
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
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
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
