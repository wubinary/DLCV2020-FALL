#!/bin/bash

## download model
bash download.sh 
wait

## run hw2-1
TEST_DIR=$1
OUT_CSV=$2

python3 p1/inference.py --test_dir $TEST_DIR --out_csv $OUT_CSV

