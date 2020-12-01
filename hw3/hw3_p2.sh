#!/bin/bash

# download model
#mkdir -p p2/result
if [ ! -f "./p2/result/100_netG.pth" ]; then
	echo " [Info] download 100_netG.pth  "
	./download_gdrive.sh 1NTTy0KSHi1NAl2ock6BLV3s9OdSaIkkC p2/result/100_netG.pth 
fi 
wait

# inference 32 images
OUT_PATH=$1
python3 p2/inference.py --out_path $OUT_PATH

