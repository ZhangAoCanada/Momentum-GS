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
        --iterations) iterations="$2"; shift ;;
        --update_until) update_until="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")
port=$(rand 10000 30000)
addr_random=$(rand 0 255)
master_port_random=$(rand 10000 30000)

python train_coarse.py --train_val_partition -s data/${data} --custom_test data/${custom_test} --images ${images} --resolution ${resolution} -m outputs/${data}/${logdir}/$time --iterations ${iterations} --update_until ${update_until}