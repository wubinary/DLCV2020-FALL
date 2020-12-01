import warnings
warnings.filterwarnings('ignore')
import wandb
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from utils import mean_
from dataset import transforms, Face_Dataset, DataLoader 
from model.my_model import ResNet18_VAE 
from model.vanilla_vae import VanillaVAE
from loss import VAE_Loss 

def _run_train(args, dataloader, model, criterion, opt):
    model.train()
    
    step_loss,step_mse,step_kld = [],[],[]
    for idx, (img) in enumerate(dataloader):
        b = img.size(0)
        
        img = img.to(args.device)
        
        img_recons, z, mean, logvar = model(img)
        
        opt.zero_grad()
        loss, mse, kld = criterion(img_recons, img, mean, logvar)
        loss.backward()
        opt.step()
        
        step_loss.append(loss.item())
        step_mse.append(mse.item())
        step_kld.append(kld.item())
        wandb.log({'train_loss':loss.item(), 'train_mse':mse.item(), 'train_kld':kld.item()})
        print('\t [{}/{}] loss:{:.3f} mse:{:.3f} kld:{:.3f}'.format(
                idx+1,
                len(dataloader),
                mean_(step_loss),
                mean_(step_mse),
                mean_(step_kld))
            , end='      \r')
        
    return mean_(step_loss), mean_(step_mse), mean_(step_kld)

@torch.no_grad()
def _run_eval(args, dataloader, model, criterion):
    model.eval()
     
    step_loss,step_mse,step_kld = [],[],[]
    for idx, (img) in enumerate(dataloader):
        b = img.size(0)
        
        img = img.to(args.device)
        
        img_recons, z, mean, logvar = model(img)
        
        loss, mse, kld = criterion(img_recons, img, mean, logvar)
        
        step_loss.append(loss.item())
        step_mse.append(mse.item())
        step_kld.append(kld.item())
        print('\t [{}/{}] loss:{:.3f} mse:{:.3f} kld:{:.3f}'.format(
                idx+1,
                len(dataloader),
                mean_(step_loss),
                mean_(step_mse),
                mean_(step_kld))
            , end='      \r')
        
    wandb.log({'valid_loss':mean_(step_loss), 'valid_mse':mean_(step_mse), 'valid_kld':mean_(step_kld)})
    return mean_(step_loss), mean_(step_mse), mean_(step_kld)

def train(args, train_dataloader, valid_dataloader):
    
    if args.model == 'vanilla_vae':
        model = VanillaVAE(3, args.latent_dim)
    elif args.model == 'resnet18_vae':
        model = ResNet18_VAE(z_dim=args.latent_dim)
    else:
        raise NotImplementedError(f'model not implemented {args.model}')
    model.to(args.device)
    
    # loss
    criterion = VAE_Loss(args.lambda_kld).to(args.device)

    # optim
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    milestones = [1,5,10,20]
    #train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2) #learning rate decay
    
    best_mse = 100
    for epoch in range(1,args.epochs+1):
        print(f"\tEpoch {epoch}")
        
        ####################### train ##########################
        mean_loss, mean_mse, mean_kld = _run_train(args, train_dataloader, model, criterion,
                                                                                optimizer)
        print("\t train loss:{:.5f} mse:{:.5f} kld:{:.5f}".format(mean_loss, mean_mse, mean_kld))
        
        ##################### validation #######################
        mean_loss, mean_mse, mean_kld = _run_eval(args, valid_dataloader, model, criterion)
        print("\t valid loss:{:.5f} mse:{:.5f} kld:{:.5f}".format(mean_loss, mean_mse, mean_kld))
        
        ##################### save model #######################
        if epoch>milestones[1] and mean_mse<best_mse:
            best_mse = mean_mse
            torch.save(model.state_dict(), f"./result/best_{args.model}.pth")
            print('\t [Info] save weights')
        if epoch in milestones:
            torch.save(model.state_dict(), f"./result/{epoch}_{args.model}.pth")
            print('\t [Info] save weights, epoch:{epoch}')

def parse_args():
    parser = argparse.ArgumentParser(description='Image Segmentation')
    parser.add_argument('--model', default='vanilla_vae',
                    choices=['resnet18_vae','vanilla_vae'],
                    help='resnet18_vae, vanilla_vae')
    parser.add_argument('--latent_dim', type=int, default=64,
                    help='z latent dimension')
    parser.add_argument('--lambda_kld', type=float, default=1e-2,
                    help='loss = mse + lambda*kld')
    parser.add_argument('--batch', type=int, default=16,
                    help='batchsize 8')
    parser.add_argument('--epochs', type=int, default=30,
                    help='epochs to train')
    parser.add_argument('--lr', type=float, default=5e-3,
                    help='learning rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                    help='cpu or cuda:0 or cuda:1')

    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    
    args = parse_args()
    
    wandb.init(config=args, project='dlcv_face_vae')

    train_transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = Face_Dataset('../hw3_data/face/train', train_transform)
    valid_dataset = Face_Dataset('../hw3_data/face/test', valid_transform)
    
    train_dataloader = DataLoader(train_dataset,
                                 batch_size=args.batch,
                                 shuffle=True,
                                 num_workers=8)
    valid_dataloader = DataLoader(valid_dataset,
                                 batch_size=args.batch*2,
                                 shuffle=False,
                                 num_workers=8)
    
    train(args, train_dataloader, valid_dataloader)

