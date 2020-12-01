SHELL := /bin/zsh 

DEVICE=cuda:0
BSIZE=128

main: problem3_1 problem3_2 problem3_3 

#######################################################

problem3_1: train_usps_2_mnistm train_mnistm_2_svhn train_svhn_2_usps 

train_usps_2_mnistm:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train3_1.py --source usps --target mnistm --bsize $(BSIZE) --warmup_epochs 5 --epochs 30 --lr 1e-3 --device $(DEVICE)

train_mnistm_2_svhn:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train3_1.py --source mnistm --target svhn --bsize $(BSIZE) --warmup_epochs 5 --epochs 30 --lr 1e-3 --device $(DEVICE)

train_svhn_2_usps:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train3_1.py --source svhn --target usps --bsize $(BSIZE) --warmup_epochs 5 --epochs 30 --lr 1e-3 --device $(DEVICE)


######################################################

problem3_2: train_dann_usps_2_mnistm train_dann_mnistm_2_svhn train_dann_svhn_2_usps 

train_dann_usps_2_mnistm:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train3_2.py --source usps --target mnistm --bsize $(BSIZE) --warmup_epochs 1 --epochs 30 --lamb 0.01 --lr 1e-3 --device $(DEVICE)

train_dann_mnistm_2_svhn:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train3_2.py --source mnistm --target svhn --bsize $(BSIZE) --warmup_epochs 1 --epochs 30 --lamb 0.01 --lr 1e-3 --device $(DEVICE) 

train_dann_svhn_2_usps:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train3_2.py --source svhn --target usps --bsize $(BSIZE) --warmup_epochs 1 --epochs 40 --lamb 0.01 --lr 1e-3 --device $(DEVICE)

#######################################################

problem3_3: train_mnistm_2_mnistm train_svhn_2_svhn train_usps_2_usps 

train_mnistm_2_mnistm:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train3_3.py --source mnistm --target mnistm --bsize $(BSIZE) --warmup_epochs 5 --epochs 30 --lr 1e-3 --device $(DEVICE)

train_svhn_2_svhn:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train3_3.py --source svhn --target svhn --bsize $(BSIZE) --warmup_epochs 5 --epochs 30 --lr 1e-3 --device $(DEVICE)

train_usps_2_usps:
	source ~/anaconda3/bin/activate && conda activate dlcv && python3 train3_3.py --source usps --target usps --bsize $(BSIZE) --warmup_epochs 5 --epochs 30 --lr 1e-3 --device $(DEVICE)





