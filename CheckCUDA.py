import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU devices found: {len(gpu_devices)}")
if gpu_devices:
    print("GPU is available")
    for device in gpu_devices:
        print(f"Device name: {device.name}")
else:
    print("GPU is not available")