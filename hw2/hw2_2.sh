#!/bin/bash

## download model
bash download.sh 
wait

## run hw2-1
TEST_DIR=$1
OUT_DIR=$2

python3 p2/inference.py --model fcn32s --model_path p2/result/ --test_dir $TEST_DIR --out_dir $OUT_DIR

