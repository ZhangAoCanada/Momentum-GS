iterations=30000
resolution=-1
test_path=matrix_city/aerial/test/block_all/
update_until=15000
exp_name=matrixcity-coarse

ulimit -n 100000

echo "@@@ Start ${exp_name} @@@"

./train_coarse.sh -d matrix_city/aerial/train/block_all/ --custom_test ${test_path} --images "input_cached" --resolution ${resolution} -l ${exp_name} --iterations ${iterations} --update_until ${update_until}