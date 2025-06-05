sudo /usr/local/cuda-12.1/bin/ncu -o profile -f \
--nvtx --nvtx-include "Base/" --nvtx-include "Shrink/" --nvtx-include "Expand/" --nvtx-include "Cudagraph/" \
--set full \
--metrics sm__warps_active,sm__cycles_active,sm__inst_executed,l1tex__m_l1tex2xbar_write_bytes \
--target-processes all \
~/anaconda3/envs/slora/bin/python $1