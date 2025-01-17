function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

gpu_list=2
gpu_num=1
addr_random=221
master_port_random=23959
# data=bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train
data=bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/train
custom_test=bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test
images=input_cached
resolution=-1
logdir=debugging
partition_name=block8_ssim0.05
block_num=8
consistency_loss_weight=50
checkpoint_tmp_dir=data/Momentum-GS/tmp/continous/
adaptive_sigma=9.0
time=debugging3
# time=$(date "+%Y-%m-%d_%H:%M:%S")
port=$(rand 10000 30000)
addr_random=$(rand 0 255)
master_port_random=$(rand 10000 30000)

CUDA_VISIBLE_DEVICES=${gpu_list} python -m torch.distributed.launch --nproc_per_node=${gpu_num} --use_env --master_addr=127.0.0.${addr_random} --master_port=${master_port_random} continous_model/train_continous.py -s data/${data} --custom_test data/${custom_test} --images ${images} --resolution ${resolution} --port $port -m outputs_continous/debug_continous/${logdir}/$time --partition_name ${partition_name} --block_num ${block_num} --aabb -3.5 -4 -10 4.5 2 10 --block_dim 2 4 1 --consistency_loss_weight ${consistency_loss_weight} --checkpoint_tmp_dir ${checkpoint_tmp_dir} --adaptive_sigma ${adaptive_sigma}