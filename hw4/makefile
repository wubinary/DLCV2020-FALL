
all: dowload test_p1 test_p2 test_p3

dowload:
	bash ./hw4_download.sh 

test_p1:
	bash ./hw4_1.sh ./hw4_data/val.csv ./hw4_data/val ./hw4_data/val_testcase.csv ./pred.csv 
	$(MAKE) valid_acc 

test_p2:
	bash ./hw4_2.sh ./hw4_data/val.csv ./hw4_data/val ./hw4_data/val_testcase.csv ./pred.csv 
	$(MAKE) valid_acc 

test_p3:
	bash ./hw4_3.sh ./hw4_data/val.csv ./hw4_data/val ./hw4_data/val_testcase.csv ./pred.csv 
	$(MAKE) valid_acc 

valid_acc:
	python3 eval.py ./pred.csv ./hw4_data/val_testcase_gt.csv

