from model import init_weights, Generator, Discriminator

import torch 
import argparse
import numpy as np 

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, 
                    default='./test_output',
                    help='output generated image path')
    args = parser.parse_args()
    return args

@torch.no_grad()
def inference(args):

    z_dim = 100
    netG = Generator(z_dim)
    netG.load_state_dict(torch.load('./p2/result/100_netG.pth',
                                    map_location='cpu'))
    netG.cuda()

    np.random.seed(87)
    latent = torch.tensor(np.random.normal(size=(32,z_dim))).float()

    out_imgs = netG(latent.cuda())/2.+.5

    fig = plt.figure(figsize=(8., 16.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(4, 8),  # creates 2x2 grid of axes
                     axes_pad=0.01,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, out_imgs.detach().cpu().numpy()):
        # Iterating over the grid returns the Axes.
        im = np.moveaxis(im,0,-1)
        ax.imshow(im)
    
    f = args.out_path
    plt.savefig(f)

    print(f" [Info] export image {f}")


args = args_parser()

inference(args)

