TEST_CSV=$1
TEST_DIR=$2
TESTCASE_CSV=$3
OUTPUT_CSV=$4

python3 p2/test_testcase.py --load ./p2/checkpoints/hallu20_shot1_trainway30_validway5_parametric_best.pth --distance parametric --test_csv $TEST_CSV --test_data_dir $TEST_DIR --testcase_csv $TESTCASE_CSV --output_csv $OUTPUT_CSV

