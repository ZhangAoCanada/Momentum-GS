gpu_num=2
gpu_list=2,3
checkpoint_tmp_dir=data/Momentum-GS/tmp/residence/
block_num=8
partition_name=residence_c20_r4
aabb="-25,-200,-270,175,200,60"
block_dim="2,1,4"
consistency_loss_weight=50
adaptive_sigma=10.0
resolution=4
train_path=VastGaussian/Residence/residence-pixsfm/train/
test_path=VastGaussian/Residence/residence-pixsfm/val/
exp_name=residence-${gpu_num}gpus-8blocks

if [ $((block_num % gpu_num)) -ne 0 ]
then
    echo "Error: block_num (${block_num}) must be divisible by gpu_num (${gpu_num})"
    exit 1
fi

# ulimit -n 100000
ulimit -n 32768

echo "@@@ Start ${exp_name} @@@"

./train.sh -d ${train_path} --custom_test ${test_path} --images "images" --resolution ${resolution} -l ${exp_name} --gpu_num ${gpu_num} --gpu_list ${gpu_list} --partition_name ${partition_name} --block_num ${block_num} --aabb ${aabb} --block_dim ${block_dim} --consistency_loss_weight ${consistency_loss_weight} --checkpoint_tmp_dir ${checkpoint_tmp_dir} --adaptive_sigma ${adaptive_sigma}