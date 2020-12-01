#!/bin/bash

# download model
#mkdir -p p2/result
if [ ! -f "./p4/result/usps2mnistm/best_model.pth" ]; then
	echo " [Info] download usps2mnistm best_model.pth  "
	./download_gdrive.sh 1Ln0LRCkPwx4EkJXfw1Z4o2VYQ-g99Vf_ p4/result/usps2mnistm/best_model.pth  
fi 
if [ ! -f "./p4/result/mnistm2svhn/best_model.pth" ]; then
	echo " [Info] download mnistm2svhn best_model.pth  "
	./download_gdrive.sh 1sFOHuChd_dNWdaRHjU01fhRNd_8MYpnr p4/result/mnistm2svhn/best_model.pth  
fi 
if [ ! -f "./p4/result/svhn2usps/best_model.pth" ]; then
	echo " [Info] download svhn2usps best_model.pth  "
	./download_gdrive.sh 1IFO6KQ76j9fStYG8QO53jc0nuSx-ZWy6 p4/result/svhn2usps/best_model.pth  
fi 
wait

# inference
IMG_PATH=$1
TARGET=$2
OUT_CSV=$3
python3 p4/inference.py --dataset_path $IMG_PATH --target $TARGET --out_csv $OUT_CSV

