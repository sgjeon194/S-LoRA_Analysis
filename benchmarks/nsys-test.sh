sudo /usr/local/cuda-12.1/bin/nsys profile -o profile \
--gpu-metrics-device=0 --cpuctxsw=none --force-overwrite true \
--trace=cuda,nvtx \
--cuda-graph-trace=node \
~/anaconda3/envs/slora/bin/python $1