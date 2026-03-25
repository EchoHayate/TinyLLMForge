import torch
import time
from tinyvllm.layers.linear import QuantLinear
from tinyvllm.kernels.quant_matmul import w8a16_gemm_fwd

def test_quant_linear():
    print("Testing QuantLinear and Triton W8A16 Kernel...")
    
    # Dimensions
    batch_size = 8
    seq_len = 512
    in_features = 4096
    out_features = 4096
    
    print(f"Shapes - Input: [{batch_size * seq_len}, {in_features}], Weight: [{out_features}, {in_features}]")
    
    # Create fake activations
    x = torch.randn((batch_size * seq_len, in_features), dtype=torch.float16, device="cuda")
    
    # Create fake INT8 weights and scales
    weight_int8 = torch.randint(-127, 127, (out_features, in_features), dtype=torch.int8, device="cuda")
    weight_scale = torch.rand((out_features,), dtype=torch.float16, device="cuda") * 0.05
    
    # Reference PyTorch Output (Dequantize then MM)
    fp16_weight = weight_int8.to(torch.float16) * weight_scale.view(-1, 1)
    
    torch.cuda.synchronize()
    start = time.time()
    ref_out = torch.nn.functional.linear(x, fp16_weight)
    torch.cuda.synchronize()
    pt_time = time.time() - start
    
    # Triton Output
    torch.cuda.synchronize()
    start = time.time()
    triton_out = w8a16_gemm_fwd(x, weight_int8, weight_scale)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    # Compare Out
    diff = torch.abs(ref_out - triton_out).max().item()
    print(f"Max absolute difference between PyTorch and Triton: {diff:.6f}")
    
    # Note: Because of different accumulation precision in kernel vs F.linear, 
    # small differences (~1e-3) are expected and normal.
    if diff < 0.05:
        print("✅ Correctness check PASSED!")
    else:
        print("❌ Correctness check FAILED! Difference is too large.")
        
    print(f"\nPerformance:")
    print(f"PyTorch Time: {pt_time * 1000:.2f} ms")
    print(f"Triton Time:  {triton_time * 1000:.2f} ms")
    
    if triton_time < pt_time:
        print("🚀 Triton kernel is FASTER!")
    else:
        print("⚠️ Triton kernel is slower (might need tuning for specific shapes)")

if __name__ == "__main__":
    test_quant_linear()
