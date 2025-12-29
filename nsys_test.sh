# Test for LoRA with different batch size

target_dir=".profiling_results"

if [ ! -d "$target_dir" ]; then
    mkdir "$target_dir"
    echo "Created directory: $target_dir"
fi


rank=64
j=1
while [ $j -le 3 ]
do
    case "$j" in
        1) distype="distinct";;
        2) distype="uniform";;
        3) distype="identical";;
    esac

    i=1
    while [ $i -le 256 ] # dispatch sgmv reports error on batch size 256 for distinct
    do
        padded_i=$(printf "%03d" $i)
        padded_rank=$(printf "%03d" $rank)
        echo "Running with batch_size=$i"
        sudo -E /usr/local/cuda/bin/nsys profile -o .profiling_results/batch_size_${padded_i}_rank_${padded_rank}_${distype} \
            --gpu-metrics-device=0 --cpuctxsw=none --force-overwrite true \
            --trace=cuda,nvtx \
            --cuda-graph-trace=node \
            .venv/bin/python benchmarks/decode_dispatch_sgmv_test.py --batch_size $i --lin 256 --lora_rank $rank --distribution $j

        i=$(( i*2 ))
        echo "============================= Finished!! ============================="
        echo ""
        echo ""
    done
    j=$(( j+1 ))
done