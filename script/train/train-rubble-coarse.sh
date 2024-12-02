iterations=30000
resolution=4
test_path=mill19/rubble-pixsfm/val/
update_until=15000
exp_name=rubble-coarse

ulimit -n 100000

echo "@@@ Start ${exp_name} @@@"

./train_coarse.sh -d mill19/rubble-pixsfm/train/ --custom_test ${test_path} --images "images" --resolution ${resolution} -l ${exp_name} --iterations ${iterations} --update_until ${update_until}