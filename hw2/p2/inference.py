import torch
import torch.nn as nn 
import torch.nn.functional as F
from PIL import Image 
import argparse

from utils import *
from dataset import *
from unet.unet_model import UNet
#from fcn.models.fcn32s import FCN32s
from model import VGG16_FCN32s, VGG16_FCN8s

@torch.no_grad()
def inference(args, dataloader):
    if str(args.model).lower()=='fcn32s':
        model = VGG16_FCN32s(n_classes=7)
        model.load_state_dict(torch.load(f'{args.model_path}/best_fcn32s.pth'))
    elif str(args.model).lower()=='fcn8s':
        model = VGG16_FCN8s(n_classes=7)
        model.load_state_dict(torch.load(f'{args.model_path}/best_fcn8s.pth'))
    else:
        model = UNet(n_channels=3, n_classes=7)
        model.load_state_dict(torch.load(f'{args.model_path}/best_unet.pth'))
    #model = nn.DataParallel(model)
    model.eval()
    model.cuda()

    for idx, (images, path) in enumerate(dataloader):
        b = images.size(0)

        predict = model(images.cuda())
        predict = F.softmax(predict.permute(0,2,3,1), dim=-1)
        predict = torch.argmax(predict, dim=-1)
        predict = predict.cpu().numpy()

        for s in range(b):
            pred_img = np.zeros((512,512,3)).astype(np.uint8)
            for c in range(len(class_map)):
                pred_img[ predict[s]==c ] = class_map[c]
            pred_img = Image.fromarray(pred_img)
            pred_img.save(path[s])
        print(f'\t[{(idx+1)*b}/{len(dataloader.dataset)}]', end='  \r')

def parse_args():
    parser = argparse.ArgumentParser(description='Image Segmentation')
    parser.add_argument('--model', type=str, default='fcn32s',
                    help='fcn32s or fcn8s or unet')
    parser.add_argument('--model_path', type=str, default='./p2/result', 
                    help='model .pth path')
    parser.add_argument('--test_dir', type=str, default='../hw2_data/p2_data/validation',
                    help='test directory')
    parser.add_argument('--out_dir', type=str, default='../hw2_data/p2_data/validation/pred',
                    help='predict out directory')

    args = parser.parse_args()
    return args 

if __name__=='__main__':

    args = parse_args()
    
    size=512
    transform = transforms.Compose([
        #transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = p2_Dataset_test(args.test_dir, args.out_dir, transform)

    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=False,
                            num_workers=8)
   
    inference(args, dataloader)

