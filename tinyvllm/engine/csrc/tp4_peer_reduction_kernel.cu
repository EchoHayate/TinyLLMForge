#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__global__ void publish_kernel(
    uint64_t* flags,
    int64_t flag_index,
    uint64_t generation) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence_system();
    reinterpret_cast<volatile uint64_t*>(flags)[flag_index] =
        generation;
  }
}

__global__ void reduce_add_residual_kernel(
    const float* rank0,
    const float* rank1,
    const float* rank2,
    const float* rank3,
    const uint64_t* flag0,
    const uint64_t* flag1,
    const uint64_t* flag2,
    const uint64_t* flag3,
    int32_t* status,
    const __nv_bfloat16* residual,
    __nv_bfloat16* output,
    int64_t rank,
    int64_t flag_index,
    uint64_t generation,
    int64_t element_count,
    uint64_t timeout_clocks) {
  __shared__ int32_t ready;
  if (threadIdx.x == 0) {
    ready = 1;
    const uint64_t started = clock64();
    const volatile uint64_t* flags[4] = {
        flag0,
        flag1,
        flag2,
        flag3,
    };
    for (int peer_rank = 0; peer_rank < 4; ++peer_rank) {
      if (peer_rank == rank) {
        continue;
      }
      while (flags[peer_rank][flag_index] != generation) {
        if (clock64() - started > timeout_clocks) {
          ready = 0;
          break;
        }
      }
      if (ready == 0) {
        break;
      }
    }
    status[0] = ready == 1 ? 0 : 1;
  }
  __syncthreads();
  if (ready == 0) {
    return;
  }
  for (
      int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
      index < element_count;
      index += blockDim.x * gridDim.x) {
    const float reduced = (
        rank0[index]
        + rank1[index]
        + rank2[index]
        + rank3[index]);
    const __nv_bfloat16 rounded = __float2bfloat16_rn(reduced);
    output[index] = __float2bfloat16_rn(
        __bfloat162float(rounded)
        + __bfloat162float(residual[index]));
  }
}

}  // namespace

void launch_tp4_publish(
    uint64_t* flags,
    int64_t flag_index,
    uint64_t generation,
    uintptr_t stream) {
  publish_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      flags,
      flag_index,
      generation);
}

void launch_tp4_reduce_add_residual(
    const float* rank0,
    const float* rank1,
    const float* rank2,
    const float* rank3,
    const uint64_t* flag0,
    const uint64_t* flag1,
    const uint64_t* flag2,
    const uint64_t* flag3,
    int32_t* status,
    const at::BFloat16* residual,
    at::BFloat16* output,
    int64_t rank,
    int64_t flag_index,
    uint64_t generation,
    int64_t element_count,
    uint64_t timeout_clocks,
    uintptr_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>(
      (element_count + threads - 1) / threads);
  reduce_add_residual_kernel<<<
      blocks,
      threads,
      0,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      rank0,
      rank1,
      rank2,
      rank3,
      flag0,
      flag1,
      flag2,
      flag3,
      status,
      reinterpret_cast<const __nv_bfloat16*>(residual),
      reinterpret_cast<__nv_bfloat16*>(output),
      rank,
      flag_index,
      generation,
      element_count,
      timeout_clocks);
}
