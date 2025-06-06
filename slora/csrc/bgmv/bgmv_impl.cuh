// adapted from https://github.com/punica-ai/punica/blob/master/punica/ops/csrc/bgmv/bgmv_impl.cuh
#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>
#include <stdio.h>
#include <nvtx3/nvToolsExt.h>
#include "vec_dtypes.cuh"

#include <vector>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <climits>

namespace cg = cooperative_groups;

__device__ __inline__ int print_kernel() {
    unsigned int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));

    return smid;
}

template <int feat_in, int feat_out, typename T>
__global__ void bgmv_multi_lora_rank_shrink_kernel(T *__restrict__ Y, const T *__restrict__ X,
                                                   const T *__restrict__ W,
                                                   const int64_t *__restrict__ start_indicies,
                                                   const int64_t *__restrict__ lora_ranks,
                                                   const int64_t *__restrict__ loc_indicies,
                                                   const int64_t *__restrict__ indicies,
                                                   int64_t qkvo)
{
    auto block = cg::this_thread_block();
    size_t j = blockIdx.x;
    size_t batch_idx = blockIdx.y;
    constexpr size_t vec_size = 16 / sizeof(T); // 항상 8이어야함
    constexpr size_t tx = 32;
    constexpr size_t ty = 4;
    constexpr size_t num_pipeline_stages = 2;
    constexpr size_t tile_size = tx * ty * vec_size;
    __shared__ T W_shared[num_pipeline_stages * tile_size];
    __shared__ T X_shared[num_pipeline_stages * tile_size];
    __shared__ float y_warpwise[ty];

    size_t lora_idx = indicies[batch_idx];
    size_t lora_rank = lora_ranks[lora_idx] / 4; // if j >= lora_rank, we do not need to do the computation
    // if (blockIdx.x == 63 && blockIdx.y == 773 && threadIdx.x == 0 && threadIdx.y == 0)
    // {
    //     printf("feat in : %d, feat out : %d\n", feat_in, feat_out);
    //     printf("tile size : %lu, vec size : %lu, type size : %lu \n", tile_size, vec_size, sizeof(T));
    //     printf("lora idx : %lu (must be smaller than lora_rank size)\n", lora_idx);
    //     printf("lora rank : %lu (must be bigger than 63)\n", lora_rank);
    // }
    unsigned int blkIdx = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int thrIdx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int out_idx = blkIdx * blockDim.x * blockDim.y + thrIdx;

    // used_sms[out_idx] = print_kernel();

    if (j >= lora_rank)
    {
        return;
    }
    size_t mem_pos = start_indicies[lora_idx] + lora_rank * qkvo;
    size_t idx = loc_indicies[mem_pos + j];

    size_t W_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
    size_t X_shared_offset[num_pipeline_stages] = {0U, 1U * tile_size};
    auto pipe = cuda::make_pipeline();

    // pipeline load W/X and compute WX;
    pipe.producer_acquire();
    // not sure if this idx*feat_in is expensive when feat_in is large
    cuda::memcpy_async(
        W_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
        W + (idx * feat_in) + (threadIdx.y * tx + threadIdx.x) * vec_size,
        cuda::aligned_size_t<16>(16), pipe);
    cuda::memcpy_async(
        X_shared + (threadIdx.y * tx + threadIdx.x) * vec_size,
        X + (batch_idx * feat_in) + (threadIdx.y * tx + threadIdx.x) * vec_size,
        cuda::aligned_size_t<16>(16), pipe);
    pipe.producer_commit();
    size_t copy_idx, compute_idx;
    float y = 0.f;
    vec_t<T, vec_size> x_vec, w_vec;
    size_t tile_idx;

#pragma unroll
    for (tile_idx = 1; tile_idx < (feat_in + tile_size - 1) / tile_size; ++tile_idx)
    {
        copy_idx = tile_idx % num_pipeline_stages;
        // pipeline stage: async copy W fragment
        pipe.producer_acquire();
        if (tile_idx * tile_size + threadIdx.y * tx * vec_size < feat_in)
        {
            cuda::memcpy_async(W_shared + W_shared_offset[copy_idx] + (threadIdx.y * tx + threadIdx.x) * vec_size,
                               W + (idx * feat_in) + tile_idx * tile_size + (threadIdx.y * tx + threadIdx.x) * vec_size,
                               cuda::aligned_size_t<16>(16), pipe);
            cuda::memcpy_async(X_shared + X_shared_offset[copy_idx] + (threadIdx.y * tx + threadIdx.x) * vec_size,
                               X + (batch_idx * feat_in) + tile_idx * tile_size + (threadIdx.y * tx + threadIdx.x) * vec_size,
                               cuda::aligned_size_t<16>(16), pipe);
        }
        pipe.producer_commit();

        compute_idx = (tile_idx - 1) % num_pipeline_stages;
        // pipeline stage: compute WX
        pipe.consumer_wait();
        block.sync();
        x_vec.load(X_shared + X_shared_offset[compute_idx] +
                   (threadIdx.y * tx + threadIdx.x) * vec_size);
        w_vec.load(W_shared + W_shared_offset[compute_idx] +
                   (threadIdx.y * tx + threadIdx.x) * vec_size);
        float sum = 0.f;
#pragma unroll
        for (size_t i = 0; i < vec_size; ++i)
        {
            sum += float(w_vec[i]) * float(x_vec[i]);
        }
#pragma unroll
        for (size_t offset = tx / 2; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        y_warpwise[threadIdx.y] = sum;
        block.sync();
#pragma unroll
        for (size_t i = 0; i < ty; ++i)
        {
            y += y_warpwise[i];
        }

        block.sync();
        pipe.consumer_release();
    }

    compute_idx = (tile_idx - 1) % num_pipeline_stages;
    // final pipeline stage
    pipe.consumer_wait();
    block.sync();
    if ((tile_idx - 1) * tile_size + (threadIdx.y * tx + threadIdx.x) * vec_size >= feat_in)
    {
        x_vec.fill(T(0.0));
        w_vec.fill(T(0.0));
    }
    else
    {
        x_vec.load(X_shared + X_shared_offset[compute_idx] +
                   (threadIdx.y * tx + threadIdx.x) * vec_size);
        w_vec.load(W_shared + W_shared_offset[compute_idx] +
                   (threadIdx.y * tx + threadIdx.x) * vec_size);
    }
    float sum = 0.f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i)
    {
        sum += float(w_vec[i]) * float(x_vec[i]);
    }
#pragma unroll
    for (size_t offset = tx / 2; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    y_warpwise[threadIdx.y] =
        ((tile_idx - 1) * tile_size + threadIdx.y * tx * vec_size < feat_in)
            ? sum
            : 0.f;
    block.sync();
#pragma unroll
    for (size_t i = 0; i < ty; ++i)
    {
        y += y_warpwise[i];
    }

    block.sync();
    pipe.consumer_release();

    // write Y;
    if (block.thread_rank() == 0)
    {
        Y[batch_idx * feat_out + j] = y;
    }
}

// nthrs = (2, 16, 4)
template <int feat_in, int feat_out, typename T>
__global__ void bgmv_multi_lora_rank_expand_kernel(T *__restrict__ Y, const T *__restrict__ X,
                                                   const T *__restrict__ W,
                                                   const int64_t *__restrict__ start_indicies,
                                                   const int64_t *__restrict__ lora_ranks,

                                                   const int64_t *__restrict__ loc_indicies,
                                                   const int64_t *__restrict__ indicies,
                                                   int64_t qkvo,
                                                   const T *__restrict__ lora_scales)
{
    auto block = cg::this_thread_block();
    size_t tile_idx = blockIdx.x;
    size_t batch_idx = blockIdx.y;
    size_t lora_idx = indicies[batch_idx];

    size_t lora_rank = lora_ranks[lora_idx] / 4;
    constexpr size_t vec_size = 16 / sizeof(T);
    constexpr size_t tx = feat_in / vec_size;
    static_assert(feat_in % vec_size == 0);
    constexpr size_t ty = 32 / tx;
    static_assert(32 % tx == 0);
    constexpr size_t tz = 4;

    size_t start_idx = start_indicies[lora_idx];
    size_t bin = (tile_idx * tz * ty * lora_rank + (threadIdx.z * ty + threadIdx.y) * lora_rank +
                  threadIdx.x * vec_size) /
                 feat_out;
    size_t bin_offset = (tile_idx * tz * ty * lora_rank + (threadIdx.z * ty + threadIdx.y) * lora_rank +
                         threadIdx.x * vec_size) %
                        feat_out;
    size_t w_outer_idx = loc_indicies[start_idx + lora_rank * qkvo + bin];
    size_t w_inner_idx = bin_offset;
    T scale = lora_scales[lora_idx];

    // load X and W
    vec_t<T, vec_size> x_vec;
    vec_t<T, vec_size> w_vec;

    if (threadIdx.x * vec_size >= lora_rank)
    {
        x_vec.fill(T(0.0));
        w_vec.fill(T(0.0));
    }
    else
    {
        x_vec.load(X + batch_idx * feat_in + threadIdx.x * vec_size);
        w_vec.load(W + (w_outer_idx * feat_out) + bin_offset);
    }

    float sum = 0.f;
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i)
    {
        sum += float(w_vec[i]) * float(x_vec[i]) * float(scale);
    }

#pragma unroll
    for (size_t offset = tx / 2; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x == 0)
    {
        Y[batch_idx * feat_out + tile_idx * (tz * ty) + threadIdx.z * ty + threadIdx.y] += sum;
    }
}


void write_data(int batch_size, int data_length, unsigned long long *data)
{
    std::string filename = "shrink_batch_" + std::to_string(batch_size) + ".txt";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "파일 열기 실패: " << filename << std::endl;
        return;
    }

    for(int i = 0; i < data_length; i++)
    {
        outfile << data[i] << "\n";
    }
    outfile << "-------------------------------------------" << "\n";
}

template <int feat_in, int feat_out, typename T>
void bgmv_kernel(T *__restrict__ Y, const T *__restrict__ X,
                 const T *__restrict__ W,
                 const int64_t *__restrict__ start_indicies,
                 const int64_t *__restrict__ lora_ranks,
                 const int64_t *__restrict__ loc_indicies,
                 const int64_t *__restrict__ indicies,
                 int64_t qkvo,
                 int64_t batch_size,
                 const T *__restrict__ lora_scales,
                 uintptr_t stream)
{

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    size_t vec_size = 16 / sizeof(T); // 항상 8
    if constexpr (feat_in < feat_out)
    {
        size_t tx = feat_in / vec_size; // 1
        size_t ty = 32 / tx;            // 32
        size_t tz = 4;
        dim3 nblks(feat_out / (ty * tz), batch_size); // (6, batch_size)
        dim3 nthrs(tx, ty, tz);                       // 항샹 셋을 곱하면 128

        // printf("expand block dim : %lu %lu %lu - thread dim : %lu %lu %lu\n", nblks.x, nblks.y, nblks.z, tx, ty, tz);
        bgmv_multi_lora_rank_expand_kernel<feat_in, feat_out><<<nblks, nthrs, 0, cuda_stream>>>(Y, X, W, start_indicies, lora_ranks, loc_indicies, indicies, qkvo, lora_scales);
        // printf("====================================\n");
    }
    else
    {
        assert(feat_in % (vec_size) == 0);
        dim3 nblks(feat_out, batch_size);
        // printf("shrink - block dim : (%lu %lu %lu) - thread dim (32, 4, 1)\n", nblks.x, nblks.y, nblks.z);
        // printf("%ld\n", stream);
        dim3 nthrs(32, 4);
        bgmv_multi_lora_rank_shrink_kernel<feat_in, feat_out><<<nblks, nthrs, 0, cuda_stream>>>(Y, X, W, start_indicies, lora_ranks, loc_indicies, indicies, qkvo);
    }
}

#define INST_BGMV(feat_in, feat_out, T)                                      \
    template void bgmv_kernel<feat_in, feat_out>(                            \
        T *__restrict__ Y, const T *__restrict__ X, const T *__restrict__ W, \
        const int64_t *__restrict__ start_indicies,                          \
        const int64_t *__restrict__ lora_ranks,                              \
        const int64_t *__restrict__ loc_indicies,                            \
        const int64_t *__restrict__ indicies, int64_t qkvo,                  \
        int64_t batch_size, const T *__restrict__ lora_scales, uint64_t stream);

#define INST_BGMV_TWOSIDE(T, narrow, wide) \
    INST_BGMV(narrow, wide, T)             \
    INST_BGMV(wide, narrow, T)
