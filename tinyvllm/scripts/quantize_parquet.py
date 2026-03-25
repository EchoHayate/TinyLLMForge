import os
import argparse
import glob
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm.auto import tqdm

def quantize_symmetric_int8(weight: torch.Tensor):
    """
    Symmetrically quantize a weight tensor to INT8 per-channel (row).
    Returns the quantized weight (int8) and the scale (float16).
    """
    # weight shape: [out_features, in_features]
    # scale needs to be computed per-row (out_features)
    
    # Get the maximum absolute value per row
    abs_max = torch.max(torch.abs(weight), dim=1, keepdim=True).values
    
    # Avoid division by zero
    abs_max = torch.clamp(abs_max, min=1e-5)
    
    # INT8 range is [-127, 127]
    scale = abs_max / 127.0
    
    # Quantize: q_weight = round(weight / scale)
    q_weight = torch.round(weight / scale)
    q_weight = torch.clamp(q_weight, -128, 127).to(torch.int8)
    
    # scale shape: [out_features, 1] -> [out_features]
    scale = scale.squeeze(1).to(torch.float16)
    
    return q_weight, scale

def is_linear_weight(name: str):
    """
    Check if the tensor is a weight of a linear layer that should be quantized.
    We quantize q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
    We generally do NOT quantize lm_head or embeddings.
    """
    if not name.endswith(".weight"):
        return False
        
    linear_suffixes = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
    
    for suffix in linear_suffixes:
        if suffix in name and "lm_head" not in name:
            return True
            
    return False

def convert_parquet_to_quantized_safetensors(input_dir: str, output_dir: str):
    """
    Reads HuggingFace parquet files, quantizes the linear weights to INT8,
    and saves the result to a safetensors file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure input_dir contains parquet files
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    if not parquet_files:
        # Sometimes people pass a directory that is meant to be loaded by datasets directly
        try:
            print(f"Loading dataset from {input_dir} (this might take a while)...")
            dataset = load_dataset("parquet", data_dir=input_dir, split="train")
        except Exception as e:
            print(f"Error loading parquet dataset: {e}")
            return
    else:
        print(f"Found {len(parquet_files)} parquet files. Loading...")
        dataset = load_dataset("parquet", data_files=parquet_files, split="train")

    tensors = {}
    
    print("Processing and quantizing weights...")
    
    # For a typical HF weight parquet, each row usually has 'name' and 'data' (bytes or similar)
    # We'll adapt based on what the dataset contains. If it's standard HuggingFace format,
    # the dataset usually yields an iterator of rows.
    
    # NOTE: Depending on exactly how your parquet was exported, the loading logic here
    # might need adjusting. If it's state_dict saved as parquet, it might have
    # 'key' and 'value' columns.
    
    # Assuming standard format: 'name' for tensor name, 'data' for the byte content or direct tensor
    for row in tqdm(dataset):
        # We need to know the specific columns in your parquet file.
        # Commonly, keys might be 'key', 'value', or 'name', 'tensor'
        name = row.get("key", row.get("name", None))
        
        if not name:
            raise ValueError(f"Could not find tensor name in parquet row. Row columns: {row.keys()}")
            
        # Extract the tensor
        # This part heavily depends on how the parquet was saved (e.g. from safetensors bytes, or raw lists)
        val = row.get("value", row.get("data", None))
        
        # If val is raw bytes (e.g. from serialized PyTorch tensor):
        if isinstance(val, bytes):
            # We might need to load it properly based on your exact format.
            # Assuming it's serialized via torch.save or similar, but often it requires Custom loading.
            import io
            try:
                # Attempt to load as standard PyTorch save format
                buffer = io.BytesIO(val)
                tensor = torch.load(buffer, map_location="cpu")
            except Exception:
                # Attempt Safetensors loading
                print(f"Failed to load {name} as standard PT. You may need specific deserialization logic depending on your parquet format.")
                continue
        else:
            # If it's a list or array
            tensor = torch.tensor(val)

        if torch.is_floating_point(tensor):
            tensor = tensor.to(torch.float16)

        if is_linear_weight(name):
            # Quantize
            q_weight, scale = quantize_symmetric_int8(tensor)
            
            # Save quantized weight and its scale
            # We rename the weight to indicate it's INT8, or keep the same name and add a scale tensor
            tensors[name] = q_weight
            tensors[name.replace(".weight", ".weight_scale")] = scale
        else:
            # Keep as is (FP16/BF16/FP32 or INTs)
            tensors[name] = tensor

    output_path = os.path.join(output_dir, "model_quantized.safetensors")
    print(f"Saving quantized model to {output_path}...")
    save_file(tensors, output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Qwen parquet weights to INT8 safetensors")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the parquet files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the quantized safetensors")
    
    args = parser.parse_args()
    convert_parquet_to_quantized_safetensors(args.input_dir, args.output_dir)
