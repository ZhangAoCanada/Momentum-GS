function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

gpu_list=3
gpu_num=1
addr_random=221
master_port_random=23959
# data=bdaibdai___MatrixCity/small_city/blockA_fusion_small/train
# custom_test=bdaibdai___MatrixCity/small_city/blockA_fusion_small/test
data=bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train
# data=bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/train
custom_test=bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test
# images=input
# resolution=1
images=input_cached
resolution=-1
logdir=mc-1gpu-smallaera-aerial-pretrained+freqfilter+interpolation
partition_name=block8_ssim0.05
block_num=8
consistency_loss_weight=50
checkpoint_tmp_dir=data/Momentum-GS/tmp/matrixcity/
adaptive_sigma=9.0
# time=wogsplat-densification-pointinterpolate-depth-color-1000-1-train200-depth1
time=gsplat-absgrad-depth1-l2-densification-pointinterpolate-depthdirect0.1-color0.1-1000-10-depth0.1-farconstrain3-scalingreg1e-2-minreg1e-1-bbg
# time=$(date "+%Y-%m-%d_%H:%M:%S")
port=$(rand 10000 30000)
addr_random=$(rand 0 255)
master_port_random=$(rand 10000 30000)

# CUDA_VISIBLE_DEVICES=${gpu_list} python -m torch.distributed.launch --nproc_per_node=${gpu_num} --use_env --master_addr=127.0.0.${addr_random} --master_port=${master_port_random} train_single_aera.py -s data/${data} --custom_test data/${custom_test} --images ${images} --resolution ${resolution} --port $port -m outputs/${data}/${logdir}/$time --partition_name ${partition_name} --block_num ${block_num} --aabb -3.5 -4 -10 4.5 2 10 --block_dim 2 4 1 --consistency_loss_weight ${consistency_loss_weight} --checkpoint_tmp_dir ${checkpoint_tmp_dir} --adaptive_sigma ${adaptive_sigma}
CUDA_VISIBLE_DEVICES=${gpu_list} python -m torch.distributed.launch --nproc_per_node=${gpu_num} --use_env --master_addr=127.0.0.${addr_random} --master_port=${master_port_random} train_single_aera_interpolation.py -s data/${data} --custom_test data/${custom_test} --images ${images} --resolution ${resolution} --port $port -m outputs/${data}/${logdir}/$time --partition_name ${partition_name} --block_num ${block_num} --aabb -3.5 -4 -10 4.5 2 10 --block_dim 2 4 1 --consistency_loss_weight ${consistency_loss_weight} --checkpoint_tmp_dir ${checkpoint_tmp_dir} --adaptive_sigma ${adaptive_sigma}