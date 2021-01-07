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

#download_gdrive $1 $2


########## Download p1 ###########
if [ ! -f "./p1/checkpoints/shot1_trainway30_validway5_parametric_epochs100_best.pth" ]; then
	mkdir -p ./p1/checkpoints 
	download_gdrive 1KXaN9ex8Erh_WGOLnjufPYGgBKAltrm8 ./p1/checkpoints/shot1_trainway30_validway5_parametric_epochs100_best.pth 
	echo "finish download p1 model"
fi 

########## Download p2 ###########
if [ ! -f "./p2/checkpoints/hallu20_shot1_trainway30_validway5_parametric_best.pth" ]; then
	mkdir -p ./p2/checkpoints 
	download_gdrive 1dukD9owFXeECSuVlP21rXDWsyD8T6n2r ./p2/checkpoints/hallu20_shot1_trainway30_validway5_parametric_best.pth
	echo "finish download p2 model"
fi 

########## Download p3 ###########
if [ ! -f "./p3/checkpoints/DTN_hallu20_shot1_trainway30_validway5_parametric_best.pth" ]; then
	mkdir -p ./p3/checkpoints 
	download_gdrive 1h-Q_wLr591kULR1qPR041UWoBlaHlJE- ./p3/checkpoints/DTN_hallu20_shot1_trainway30_validway5_parametric_best.pth
	echo "finish download p3 model"
fi

