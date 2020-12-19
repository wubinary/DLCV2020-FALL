
all: hw2_1 hw2_2 iou_2_2 hw2_2_best iou_2_2_best 

hw2_1:
	bash hw2_1.sh ./hw2_data/p1_data/val_50 ./

hw2_2:
	bash hw2_2.sh ./hw2_data/p2_data/validation ./hw2_data/p2_data/validation/pred 

iou_2_2:
	python3 mean_iou_evaluate.py -g hw2_data/p2_data/validation -p hw2_data/p2_data/validation/pred 

hw2_2_best:
	bash hw2_2_best.sh ./hw2_data/p2_data/validation ./hw2_data/p2_data/validation/pred_best

iou_2_2_best:
	python3 mean_iou_evaluate.py -g hw2_data/p2_data/validation -p hw2_data/p2_data/validation/pred_best


