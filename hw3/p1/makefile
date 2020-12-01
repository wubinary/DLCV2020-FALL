SHELL := /bin/zsh 

DEVICE=cuda:1

train_vanilla_vae:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train.py --model vanilla_vae --latent_dim 64 --lambda_kld 0.01 --batch 128 --epochs 20 --lr 5e-3 --device $(DEVICE)

#### resnet18 效果不好 ##########
#train_resnet18_vae:
#	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train.py --model resnet18_vae --lambda_kld 0.01 --batch 128 --epochs 20 --lr 5e-3 --device $(DEVICE)


