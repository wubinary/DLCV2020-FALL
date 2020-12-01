#!/bin/bash

# download model
#mkdir -p p1/result
if [ ! -f "./p1/result/best_vanilla_vae.pth" ]; then
	echo " [Info] download best_vanilla_vae.pth "
	./download_gdrive.sh 1F0tTxbPD9Ty9CbB4qVDREF1oxmOW8P-a p1/result/best_vanilla_vae.pth 
fi 
wait

# inference 32 images
OUT_PATH=$1
python3 p1/inference.py --out_path $OUT_PATH

