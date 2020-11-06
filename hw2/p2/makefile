SHELL := /bin/zsh 

main: train inference 

dataset:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 dataset.py
	
model:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 model.py
	
train_fcn32s:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 train.py --model fcn32s --batch 8 --epochs 50 --device cuda:0

train_fcn8s:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 train.py --model fcn8s --epochs 80 --device cuda:0 

train_unet:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 train.py --model unet --batch 4 --epochs 100
		
inference_fcn32s:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 inference.py --model fcn32s 

inference_fcn8s:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 inference.py --model fcn8s 

inference_unet:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 inference.py --model unet 

iou:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 mean_iou_evaluate.py -g ../hw2_data/p2_data/validation -p ../hw2_data/p2_data/validation/pred 

test:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 test.py

viz_mask:
	source ~/anaconda3/bin/activate && conda activate dlcv_hw2 && python3 ../viz_mask.py --img_path ../hw2_data/p2_data/validation/0010_sat.jpg --seg_path ../hw2_data/p2_data/validation/pred/0010_mask.png
