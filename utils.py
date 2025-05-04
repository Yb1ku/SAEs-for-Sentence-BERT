import pynvml
import torch
import time



def wait_for_gpu_cooldown(threshold=75, cooldown=60, check_interval=10, verbose=False):
    """
    Waits for the GPU temperature to cool down to a specified level.
    """
    if not torch.cuda.is_available():
        return

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        if temp >= threshold:
            start_time = time.time()
            if verbose:
                print(f"⚠️ GPU too hot ({temp}°C). Cooling down to ≤ {cooldown}°C...")

            while temp > cooldown:
                time.sleep(check_interval)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                if verbose:
                    print(f"   ↳ Still hot: {temp}°C... waiting...")

            if verbose:
                elapsed = int(time.time() - start_time)
                print(f"✅ GPU cooled down to {temp}°C after {elapsed} seconds.")

    except Exception as e:
        if verbose:
            print(f"[WARN] Could not get GPU temperature: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass