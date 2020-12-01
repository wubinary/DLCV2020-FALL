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

download_gdrive $1 $2
