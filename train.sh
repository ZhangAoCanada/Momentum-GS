function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        --custom_test) custom_test="$2"; shift ;;
        --images) images="$2"; shift ;;
        --resolution) resolution="$2"; shift ;;
        --gpu_num) gpu_num="$2"; shift ;;
        --gpu_list) gpu_list="$2"; shift ;;
        --block_num) block_num="$2"; shift ;;
        --partition_name) partition_name="$2"; shift ;;
        --aabb) IFS=',' read -r -a aabb <<< "$2"; shift ;;
        --block_dim) IFS=',' read -r -a block_dim <<< "$2"; shift ;;
        --checkpoint_tmp_dir) checkpoint_tmp_dir="$2"; shift ;;
        --consistency_loss_weight) consistency_loss_weight="$2"; shift ;;
        --adaptive_sigma) adaptive_sigma="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")
port=$(rand 10000 30000)
addr_random=$(rand 0 255)
master_port_random=$(rand 10000 30000)

CUDA_VISIBLE_DEVICES=${gpu_list} python -m torch.distributed.launch --nproc_per_node=${gpu_num} --use_env --master_addr=127.0.0.${addr_random} --master_port=${master_port_random} train.py -s data/${data} --custom_test data/${custom_test} --images ${images} --resolution ${resolution} --port $port -m outputs/${data}/${logdir}/$time --partition_name ${partition_name} --block_num ${block_num} --aabb "${aabb[@]}" --block_dim "${block_dim[@]}" --consistency_loss_weight ${consistency_loss_weight} --checkpoint_tmp_dir ${checkpoint_tmp_dir} --adaptive_sigma ${adaptive_sigma}