gpu_num=2
gpu_list=2,3
checkpoint_tmp_dir=data/Momentum-GS/tmp/matrixcity/
block_num=8
partition_name=block8_ssim0.05
aabb="-3.5,-4,-10,4.5,2,10"
block_dim="2,4,1"
consistency_loss_weight=50
adaptive_sigma=9.0
resolution=-1
train_path=bdaibdai___MatrixCity/small_city/aerial/train/block_all
test_path=bdaibdai___MatrixCity/small_city/aerial/test/block_all
exp_name=matrixcity-${gpu_num}gpus-8blocks

if [ $((block_num % gpu_num)) -ne 0 ]
then
    echo "Error: block_num (${block_num}) must be divisible by gpu_num (${gpu_num})"
    exit 1
fi

# ulimit -n 100000
ulimit -n 32768

echo "@@@ Start ${exp_name} @@@"

./train.sh -d ${train_path} --custom_test ${test_path} --images "input_cached" --resolution ${resolution} -l ${exp_name} --gpu_num ${gpu_num} --gpu_list ${gpu_list} --partition_name ${partition_name} --block_num ${block_num} --aabb ${aabb} --block_dim ${block_dim} --consistency_loss_weight ${consistency_loss_weight} --checkpoint_tmp_dir ${checkpoint_tmp_dir} --adaptive_sigma ${adaptive_sigma}
