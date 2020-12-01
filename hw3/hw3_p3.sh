#!/bin/bash

# download model
#mkdir -p p2/result
if [ ! -f "./p3/result/3_2/usps2mnistm/best_model.pth" ]; then
	echo " [Info] download usps2mnistm best_model.pth  "
	./download_gdrive.sh 19BpLjj1sAT-icgQjdUpwwRaRVlhW9L3x p3/result/3_2/usps2mnistm/best_model.pth  
fi 
if [ ! -f "./p3/result/3_2/mnistm2svhn/best_model.pth" ]; then
	echo " [Info] download mnistm2svhn best_model.pth  "
	./download_gdrive.sh 1hm7269PmmnKikprfCpYSGmc6vZv7hKGH p3/result/3_2/mnistm2svhn/best_model.pth  
fi 
if [ ! -f "./p3/result/3_2/svhn2usps/best_model.pth" ]; then
	echo " [Info] download svhn2usps best_model.pth  "
	./download_gdrive.sh 19lqpsxvdR5d2EWL8e-AKvdND_PlSAdBy p3/result/3_2/svhn2usps/best_model.pth  
fi 
wait

# inference
IMG_PATH=$1
TARGET=$2
OUT_CSV=$3
python3 p3/inference.py --dataset_path $IMG_PATH --target $TARGET --out_csv $OUT_CSV

