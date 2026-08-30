#include <torch/extension.h>

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

struct PeerMapping {
  void* slot_base = nullptr;
  void* flag_base = nullptr;
  int64_t peer_rank = -1;
  int64_t layer_count = 0;
  int64_t ring_size = 0;
  int64_t max_active_tokens = 0;
  int64_t hidden_size = 0;
};

std::mutex mapping_mutex;
std::unordered_map<int64_t, PeerMapping> mappings;
int64_t next_mapping_id = 1;

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(
      status == cudaSuccess,
      operation,
      " failed: ",
      cudaGetErrorString(status));
}

cudaIpcMemHandle_t decode_handle(
    const py::bytes& encoded,
    const char* name) {
  const std::string bytes = encoded;
  TORCH_CHECK(
      bytes.size() == sizeof(cudaIpcMemHandle_t),
      name,
      " must contain exactly ",
      sizeof(cudaIpcMemHandle_t),
      " bytes");
  cudaIpcMemHandle_t handle;
  std::memcpy(&handle, bytes.data(), sizeof(handle));
  return handle;
}

PeerMapping mapping_for(int64_t mapping_id) {
  std::lock_guard<std::mutex> guard(mapping_mutex);
  const auto iterator = mappings.find(mapping_id);
  TORCH_CHECK(iterator != mappings.end(), "unknown peer mapping");
  return iterator->second;
}

void validate_cuda_tensor(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void launch_tp4_publish(
    uint64_t* flags,
    int64_t flag_index,
    uint64_t generation,
    uintptr_t stream);

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
    uintptr_t stream);

py::bytes export_ipc_handle(const torch::Tensor& tensor) {
  validate_cuda_tensor(tensor, "tensor");
  cudaIpcMemHandle_t handle;
  check_cuda(
      cudaIpcGetMemHandle(&handle, tensor.data_ptr()),
      "cudaIpcGetMemHandle");
  return py::bytes(
      reinterpret_cast<const char*>(&handle),
      sizeof(handle));
}

int64_t open_mapping(
    const py::bytes& slot_handle_bytes,
    const py::bytes& flag_handle_bytes,
    const std::vector<int64_t>& slot_shape,
    const std::vector<int64_t>& flag_shape,
    int64_t peer_rank) {
  TORCH_CHECK(slot_shape.size() == 4, "slot_shape must have rank 4");
  TORCH_CHECK(flag_shape.size() == 2, "flag_shape must have rank 2");
  TORCH_CHECK(slot_shape[1] == 2, "slot ring size must be 2");
  TORCH_CHECK(flag_shape[0] == slot_shape[0], "layer count mismatch");
  TORCH_CHECK(flag_shape[1] == slot_shape[1], "ring size mismatch");
  TORCH_CHECK(peer_rank >= 0 && peer_rank < 4, "peer_rank is invalid");

  const auto slot_handle = decode_handle(
      slot_handle_bytes,
      "slot_handle");
  const auto flag_handle = decode_handle(
      flag_handle_bytes,
      "flag_handle");
  PeerMapping mapping;
  check_cuda(
      cudaIpcOpenMemHandle(
          &mapping.slot_base,
          slot_handle,
          cudaIpcMemLazyEnablePeerAccess),
      "cudaIpcOpenMemHandle(slot)");
  try {
    check_cuda(
        cudaIpcOpenMemHandle(
            &mapping.flag_base,
            flag_handle,
            cudaIpcMemLazyEnablePeerAccess),
        "cudaIpcOpenMemHandle(flag)");
  } catch (...) {
    cudaIpcCloseMemHandle(mapping.slot_base);
    throw;
  }
  mapping.peer_rank = peer_rank;
  mapping.layer_count = slot_shape[0];
  mapping.ring_size = slot_shape[1];
  mapping.max_active_tokens = slot_shape[2];
  mapping.hidden_size = slot_shape[3];

  std::lock_guard<std::mutex> guard(mapping_mutex);
  const int64_t mapping_id = next_mapping_id++;
  mappings.emplace(mapping_id, mapping);
  return mapping_id;
}

void close_mapping(int64_t mapping_id) {
  PeerMapping mapping;
  {
    std::lock_guard<std::mutex> guard(mapping_mutex);
    const auto iterator = mappings.find(mapping_id);
    if (iterator == mappings.end()) {
      return;
    }
    mapping = iterator->second;
    mappings.erase(iterator);
  }
  check_cuda(
      cudaIpcCloseMemHandle(mapping.flag_base),
      "cudaIpcCloseMemHandle(flag)");
  check_cuda(
      cudaIpcCloseMemHandle(mapping.slot_base),
      "cudaIpcCloseMemHandle(slot)");
}

void release_owned() {
}

void publish(
    const torch::Tensor& local_slot,
    const torch::Tensor& local_flags,
    int64_t layer_index,
    int64_t slot_index,
    uint64_t generation,
    uintptr_t stream) {
  validate_cuda_tensor(local_slot, "local_slot");
  validate_cuda_tensor(local_flags, "local_flags");
  TORCH_CHECK(local_slot.scalar_type() == at::kFloat, "local_slot must be FP32");
  TORCH_CHECK(local_flags.element_size() == 8, "local_flags must be uint64");
  TORCH_CHECK(local_flags.dim() == 2, "local_flags must have rank 2");
  TORCH_CHECK(layer_index >= 0 && layer_index < local_flags.size(0),
              "layer_index is invalid");
  TORCH_CHECK(slot_index >= 0 && slot_index < local_flags.size(1),
              "slot_index is invalid");
  const int64_t flag_index = (
      layer_index * local_flags.size(1) + slot_index);
  launch_tp4_publish(
      static_cast<uint64_t*>(local_flags.data_ptr()),
      flag_index,
      generation,
      stream);
}

int64_t reduce_add_residual(
    const std::vector<int64_t>& mapping_ids,
    const torch::Tensor& local_slot,
    const torch::Tensor& local_flags,
    const torch::Tensor& residual,
    const torch::Tensor& output,
    const torch::Tensor& status,
    int64_t rank,
    int64_t layer_index,
    int64_t slot_index,
    uint64_t generation,
    int64_t active_tokens,
    int64_t hidden_size,
    uint64_t timeout_clocks,
    uintptr_t stream) {
  validate_cuda_tensor(local_slot, "local_slot");
  validate_cuda_tensor(local_flags, "local_flags");
  validate_cuda_tensor(residual, "residual");
  validate_cuda_tensor(output, "output");
  validate_cuda_tensor(status, "status");
  TORCH_CHECK(mapping_ids.size() == 3, "three peer mappings are required");
  TORCH_CHECK(rank >= 0 && rank < 4, "rank is invalid");
  TORCH_CHECK(local_slot.scalar_type() == at::kFloat,
              "local_slot must be FP32");
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16,
              "residual must be BF16");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16,
              "output must be BF16");
  TORCH_CHECK(status.scalar_type() == at::kInt,
              "status must be int32");
  TORCH_CHECK(status.numel() == 1, "status must contain one element");
  TORCH_CHECK(local_slot.numel() == active_tokens * hidden_size,
              "local_slot element count mismatch");
  TORCH_CHECK(residual.numel() == local_slot.numel(),
              "residual element count mismatch");
  TORCH_CHECK(output.numel() == local_slot.numel(),
              "output element count mismatch");

  std::array<const float*, 4> slot_pointers{};
  std::array<const uint64_t*, 4> flag_pointers{};
  slot_pointers[rank] = static_cast<const float*>(local_slot.data_ptr());
  flag_pointers[rank] = static_cast<const uint64_t*>(
      local_flags.data_ptr());
  for (const auto mapping_id : mapping_ids) {
    const auto mapping = mapping_for(mapping_id);
    TORCH_CHECK(
        mapping.hidden_size == hidden_size,
        "peer hidden size mismatch");
    TORCH_CHECK(
        active_tokens <= mapping.max_active_tokens,
        "peer active-token capacity mismatch");
    const int64_t slot_offset = (
        ((layer_index * mapping.ring_size + slot_index)
         * mapping.max_active_tokens)
        * mapping.hidden_size);
    slot_pointers[mapping.peer_rank] = (
        static_cast<const float*>(mapping.slot_base) + slot_offset);
    flag_pointers[mapping.peer_rank] = static_cast<const uint64_t*>(
        mapping.flag_base);
  }
  for (int peer_rank = 0; peer_rank < 4; ++peer_rank) {
    TORCH_CHECK(slot_pointers[peer_rank] != nullptr,
                "peer slot inventory is incomplete");
    TORCH_CHECK(flag_pointers[peer_rank] != nullptr,
                "peer flag inventory is incomplete");
  }
  const int64_t flag_index = (
      layer_index * local_flags.size(1) + slot_index);
  launch_tp4_reduce_add_residual(
      slot_pointers[0],
      slot_pointers[1],
      slot_pointers[2],
      slot_pointers[3],
      flag_pointers[0],
      flag_pointers[1],
      flag_pointers[2],
      flag_pointers[3],
      static_cast<int32_t*>(status.data_ptr()),
      static_cast<const at::BFloat16*>(residual.data_ptr()),
      static_cast<at::BFloat16*>(output.data_ptr()),
      rank,
      flag_index,
      generation,
      local_slot.numel(),
      timeout_clocks,
      stream);
  check_cuda(cudaGetLastError(), "tp4 reduce kernel launch");
  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("export_ipc_handle", &export_ipc_handle);
  module.def("open_mapping", &open_mapping);
  module.def("close_mapping", &close_mapping);
  module.def("release_owned", &release_owned);
  module.def("publish", &publish);
  module.def("reduce_add_residual", &reduce_add_residual);
}
