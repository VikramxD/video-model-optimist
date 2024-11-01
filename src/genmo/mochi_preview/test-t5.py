import torch
import pynvml
from transformers import T5EncoderModel
import bitsandbytes as bnb
import time

# Constants
T5_MODEL = "google/t5-v1_1-xxl"
GPU_ID = 0  # Adjust this if you have multiple GPUs

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_ID)

def print_gpu_usage(model_type):
    # GPU memory and utilization
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

    print(f"\n--- {model_type} Model ---")
    print(f"GPU Memory Used: {memory_info.used / 1024**2:.2f} MB")
    print(f"GPU Utilization: {utilization}%")

def load_original_model():
    print("Loading original model...")
    model = T5EncoderModel.from_pretrained(T5_MODEL).to(f"cuda:{GPU_ID}")
    return model

def load_quantized_model():
    print("Loading quantized model...")
    model = T5EncoderModel.from_pretrained(T5_MODEL, load_in_8bit=True, device_map="auto")
    return model

def test_inference(model):
    # Dummy input for inference
    input_ids = torch.ones((1, 10), dtype=torch.long).to(f"cuda:{GPU_ID}")

    # Run inference and measure time
    start = time.time()
    with torch.no_grad():
        _ = model(input_ids)
    torch.cuda.synchronize()
    print(f"Inference time: {time.time() - start:.4f} seconds")

def main():
    # Original model testing
    original_model = load_original_model()
    print_gpu_usage("Original")
    test_inference(original_model)
    print_gpu_usage("Original (After Inference)")
    del original_model
    torch.cuda.empty_cache()

    # Quantized model testing
    quantized_model = load_quantized_model()
    print_gpu_usage("Quantized")
    test_inference(quantized_model)
    print_gpu_usage("Quantized (After Inference)")
    del quantized_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
