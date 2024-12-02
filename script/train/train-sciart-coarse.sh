iterations=30000
resolution=4
test_path=urbanscene3d/sci-art/val/
update_until=15000
exp_name=sciart-coarse

ulimit -n 100000

echo "@@@ Start ${exp_name} @@@"

./train_coarse.sh -d urbanscene3d/sci-art/train/ --custom_test ${test_path} --images "images" --resolution ${resolution} -l ${exp_name} --iterations ${iterations} --update_until ${update_until}