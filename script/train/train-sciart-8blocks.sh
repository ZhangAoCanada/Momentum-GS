gpu_num=2
gpu_list=2,3
checkpoint_tmp_dir=data/Momentum-GS/tmp/sciart/
block_num=8
partition_name=block8_ssim0.03
aabb="-110,-500,-205,55,100,90"
block_dim="2,1,4"
consistency_loss_weight=100
adaptive_sigma=13.0
resolution=4
train_path=VastGaussian/Sci-Art/sci-art-pixsfm/train/
test_path=VastGaussian/Sci-Art/sci-art-pixsfm/val/
exp_name=sciart-${gpu_num}gpus-8blocks

if [ $((block_num % gpu_num)) -ne 0 ]
then
    echo "Error: block_num (${block_num}) must be divisible by gpu_num (${gpu_num})"
    exit 1
fi

# ulimit -n 100000
ulimit -n 32768

echo "@@@ Start ${exp_name} @@@"

./train.sh -d ${train_path} --custom_test ${test_path} --images "images" --resolution ${resolution} -l ${exp_name} --gpu_num ${gpu_num} --gpu_list ${gpu_list} --partition_name ${partition_name} --block_num ${block_num} --aabb ${aabb} --block_dim ${block_dim} --consistency_loss_weight ${consistency_loss_weight} --checkpoint_tmp_dir ${checkpoint_tmp_dir} --adaptive_sigma ${adaptive_sigma}