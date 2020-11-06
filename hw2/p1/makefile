SHELL := /bin/zsh 

main: train inference 

train:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 train.py --epochs 15 --device cuda:1 

inference:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 inference.py 


