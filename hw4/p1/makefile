all: train_p1 test_p1 

########################################################################################

train_p1: p1_1 p1_2 p1_3 

p1_1:
	python3 train.py --shot 1 --query 15 --train_way 30 --valid_way 5 --distance euclidian 
	python3 train.py --shot 1 --query 15 --train_way 30 --valid_way 5 --distance cosine 
	python3 train.py --shot 1 --query 15 --train_way 30 --valid_way 5 --distance parametric
	
p1_2:
	python3 train.py --shot 1 --query 15 --train_way 5 --valid_way 5 --distance euclidian --epochs 300 --device cuda:1
	python3 train.py --shot 1 --query 15 --train_way 5 --valid_way 5 --distance cosine --epochs 300 --device cuda:1
	python3 train.py --shot 1 --query 15 --train_way 5 --valid_way 5 --distance parametric --epochs 300 --device cuda:1

p1_3::
	python3 train.py --shot 1 --query 15 --train_way 5 --valid_way 5 --distance parametric --epochs 300 --device cuda:1
	python3 train.py --shot 5 --query 15 --train_way 5 --valid_way 5 --distance parametric --epochs 300 --device cuda:1
	python3 train.py --shot 10 --query 15 --train_way 5 --valid_way 5 --distance parametric --epochs 300 --device cuda:1

########################################################################################

test_p1: test_1_1 test_1_2 test_1_3 

test_1_1:
	python3 test_testcase.py --load ./checkpoints/shot1_trainway30_validway5_euclidian_epochs100_best.pth --distance euclidian
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 
	python3 test_testcase.py --load ./checkpoints/shot1_trainway30_validway5_cosine_epochs100_best.pth --distance cosine 
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 
	python3 test_testcase.py --load ./checkpoints/shot1_trainway30_validway5_parametric_epochs100_best.pth --distance parametric 
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 
 
test_1_2:
	python3 test_testcase.py --load ./checkpoints/shot1_trainway5_validway5_euclidian_epochs300_best.pth --distance euclidian
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 
	python3 test_testcase.py --load ./checkpoints/shot1_trainway5_validway5_cosine_epochs300_best.pth --distance cosine 
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 
	python3 test_testcase.py --load ./checkpoints/shot1_trainway5_validway5_parametric_epochs300_best.pth --distance parametric 
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 
 
test_1_3:
	python3 test_testcase.py --load ./checkpoints/shot1_trainway5_validway5_parametric_epochs300_best.pth --distance parametric 
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 
	python3 test_testcase.py --load ./checkpoints/shot5_trainway5_validway5_parametric_epochs300_best.pth --distance parametric 
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 
	python3 test_testcase.py --load ./checkpoints/shot10_trainway5_validway5_parametric_epochs300_best.pth --distance parametric 
	python3 ../eval.py ./pred.csv ../hw4_data/val_testcase_gt.csv 

