
all: train_p2 test_p2 

train_p2: p2_1 p2_2 p2_3 

p2_1:
	python3 train.py --hallu_m 20 --shot 1 --query 15 --train_way 30 --valid_way 5 --distance parametric
p2_2:
	sleep 1
p2_3::
	python3 train.py --hallu_m 10 --shot 1 --query 15 --train_way 30 --valid_way 5 --distance parametric
	python3 train.py --hallu_m 50 --shot 1 --query 15 --train_way 30 --valid_way 5 --distance parametric 
	python3 train.py --hallu_m 100 --shot 1 --query 15 --train_way 30 --valid_way 5 --distance parametric 
 
test_p2: t2_1 t2_2 t2_3

t2_1:
	python3 test_testcase.py --load ./checkpoints/hallu20_shot1_trainway30_validway5_parametric_best.pth --distance parametric
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 

t2_2:
	sleep 1

t2_3:
	python3 test_testcase.py --load ./checkpoints/hallu10_shot1_trainway30_validway5_parametric_best.pth --distance parametric 
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 
	python3 test_testcase.py --load ./checkpoints/hallu50_shot1_trainway30_validway5_parametric_best.pth --distance parametric 
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 
	python3 test_testcase.py --load ./checkpoints/hallu100_shot1_trainway30_validway5_parametric_best.pth --distance parametric 
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 

