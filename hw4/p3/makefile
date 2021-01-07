all: p3_1 t3_1 

p3_1:
	python3 train.py --hallu_m 20 --shot 1 --query 15 --train_way 30 --valid_way 5 --distance parametric
 
t3_1:
	python3 test_testcase.py --load ./checkpoints/DTN_hallu20_shot1_trainway30_validway5_parametric_best.pth --distance parametric 
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 
	
