import pynvml
import torch
import time



def wait_for_gpu_cooldown(threshold=75, cooldown=60):
    """Pauses the execution if the GPU temperature exceeds the threshold."""
    if torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            if temp >= threshold:
                print(f"⚠️ GPU too hot ({temp}°C). Waiting untill it reaches {cooldown}°C...")
                while True:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    if temp <= cooldown:
                        break
                    time.sleep(10)
            print(f"GPU current temperature: {temp}°C")
        except Exception as e:
            print(f"Could not obtain GPU's temperature: {e}")
            print("Skipping temperature checking...")
        finally:
            try:
                pynvml.nvmlShutdown()
            except:
                pass