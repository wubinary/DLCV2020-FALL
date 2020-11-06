# Download dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12RIwW5jN1vrIn8ap1V4-m567I9tUCCBT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12RIwW5jN1vrIn8ap1V4-m567I9tUCCBT" -O hw2_data.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
mkdir hw2_data
unzip ./hw2_data.zip -d hw2_data

# Remove the downloaded zip file
rm ./hw2_data.zip
