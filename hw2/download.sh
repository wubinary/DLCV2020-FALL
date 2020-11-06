#!/bin/bash

download_gdrive(){
	fileid=$1
	filename=$2
	
	wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
		     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

	wget --load-cookies cookies.txt -O $filename \
		     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
	rm cookies.txt confirm.txt
}

##########################
########  hw2-1   ########
mkdir -p p1/result
if [ ! -f "p1/result/best.pt" ]; then
	echo " [Info] download early model "
	download_gdrive 1FVA2HKENgI5eR-JQBqtK3PT49jflmjXv p1/result/best.pt 
fi 

##########################
########  hw2-2   ########
mkdir -p p2/result
if [ ! -f "p2/result/best_fcn32s.pth" ]; then
	echo " [Info] download fcn32s model "
	download_gdrive 1X8PpaT84mksz670iGQbOJSMxVFIIklT6 p2/result/best_fcn32s.pth 
fi 

##########################
######  hw2-2 best  ######
if [ ! -f "p2/result/best_fcn8s.pth" ]; then 
	echo " [Info] download fcn8s model "
	download_gdrive 17ESPR60x_i1XxTz1xGB_y9ekUjFk95Dy p2/result/best_fcn8s.pth
fi 

