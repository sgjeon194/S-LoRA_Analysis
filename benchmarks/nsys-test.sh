sudo /usr/local/cuda/bin/nsys profile -o profile_nsys --gpu-metrics-device=0 --cpuctxsw=none --force-overwrite true --trace=cuda,nvtx ~/anaconda3/envs/slora/bin/python $1
