sudo -E /usr/local/cuda/bin/nsys profile -o .profiling_results/single_batch_size_128_lin_256_distinct \
    --gpu-metrics-device=0 --cpuctxsw=none --force-overwrite true \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node \
    .venv/bin/python benchmarks/decode_dispatch_sgmv_test.py --batch_size 128 --lin 256 --distribution 1