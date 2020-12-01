SHELL := /bin/zsh 

DEVICE=cuda:0
BSIZE=64
EPOCHS=100
LR=1e-2

problem4: train_sbada_usps_2_mnistm train_sbada_mnistm_2_svhn train_sbada_svhn_2_usps 

train_sbada_usps_2_mnistm:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train.py --source usps --target mnistm --bsize 32 --epochs 10 --lamb 1.0 --flag symnet --lr 1e-2 --arch resnet18 --device $(DEVICE)

train_sbada_mnistm_2_svhn:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train.py --source mnistm --target svhn --bsize 32 --warmup_epochs 1 --epochs 10 --lamb 1.0 --flag symnet --lr 1e-2 --arch resnet18 --device $(DEVICE) 

train_sbada_svhn_2_usps:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train.py --source svhn --target usps --bsize 64 --warmup_epochs 1 --epochs 35 --lamb 1.0 --flag test --lr 1e-2 --arch resnet18 --device $(DEVICE)


