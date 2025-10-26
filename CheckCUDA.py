#import tensorflow as tf
import torch

"""
print(f"TensorFlow version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU devices found: {len(gpu_devices)}")
if gpu_devices:
    print("GPU is available")
    for device in gpu_devices:
        print(f"Device name: {device.name}")
else:
    print("GPU is not available")
"""

# GPU (CUDA) が利用可能か確認
is_cuda_available = torch.cuda.is_available()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {is_cuda_available}")

if is_cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")