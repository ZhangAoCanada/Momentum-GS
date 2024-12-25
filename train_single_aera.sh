function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

gpu_list=3
gpu_num=1
addr_random=222
master_port_random=23958
data=bdaibdai___MatrixCity/small_city/blockA_fusion_small/train
custom_test=bdaibdai___MatrixCity/small_city/blockA_fusion_small/test
images=input_cached
resolution=-1
logdir=mc-1gpu-smallaera
partition_name=block8_ssim0.05
block_num=8
consistency_loss_weight=50
checkpoint_tmp_dir=data/Momentum-GS/tmp/matrixcity/
adaptive_sigma=9.0
time=$(date "+%Y-%m-%d_%H:%M:%S")
port=$(rand 10000 30000)
addr_random=$(rand 0 255)
master_port_random=$(rand 10000 30000)

CUDA_VISIBLE_DEVICES=${gpu_list} python -m torch.distributed.launch --nproc_per_node=${gpu_num} --use_env --master_addr=127.0.0.${addr_random} --master_port=${master_port_random} train_single_aera.py -s data/${data} --custom_test data/${custom_test} --images ${images} --resolution ${resolution} --port $port -m outputs/${data}/${logdir}/$time --partition_name ${partition_name} --block_num ${block_num} --aabb -3.5 -4 -10 4.5 2 10 --block_dim 2 4 1 --consistency_loss_weight ${consistency_loss_weight} --checkpoint_tmp_dir ${checkpoint_tmp_dir} --adaptive_sigma ${adaptive_sigma}