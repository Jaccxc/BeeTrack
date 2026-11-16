"""Pin the §116 CPU worker before importing model or numerical modules."""
import ctypes
import os
from pathlib import Path
import re


def configure_cpu(threads=2):
    """Run once in the worker, before importing the predictor or NumPy."""
    if threads not in (1, 2):
        raise ValueError("Use one or two CPU threads for this pilot")
    import psutil

    process = psutil.Process()
    allowed = sorted(process.cpu_affinity())
    process.cpu_affinity(allowed[:threads])
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(threads)

    import cv2
    import torch
    from ultralytics import utils
    from ultralytics.utils import torch_utils

    # CPU select_device() uses the imported copy in torch_utils.
    utils.NUM_THREADS = torch_utils.NUM_THREADS = threads
    os.environ["NUMEXPR_MAX_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    return cpu_readback(threads)


def cpu_readback(threads):
    """Read the effective process affinity, ATen/MKL, OpenCV and loaded OpenBLAS pools."""
    import cv2
    import psutil
    import torch
    from ultralytics import utils
    from ultralytics.utils import torch_utils

    process = psutil.Process()
    readback = dict(affinity=sorted(process.cpu_affinity()), torch=torch.get_num_threads(),
                    interop=torch.get_num_interop_threads(), opencv=cv2.getNumThreads(),
                    ultra_threads=utils.NUM_THREADS, ultra_select_device_threads=torch_utils.NUM_THREADS)
    info = torch.__config__.parallel_info()
    pools = {name:int(value) for name,value in re.findall(r"(omp|mkl)_get_max_threads\(\)\s*:\s*(\d+)",info)}
    for lib in process.memory_maps():
        name = Path(lib.path).name
        if "openblas" not in name.lower():
            continue
        dll = ctypes.CDLL(lib.path)
        for symbol in ("scipy_openblas_get_num_threads", "scipy_openblas_get_num_threads64_",
                       "openblas_get_num_threads", "openblas_get_num_threads64_"):
            if hasattr(dll, symbol):
                pools[name] = getattr(dll, symbol)()
                break
        else:
            raise RuntimeError(f"Cannot verify OpenBLAS thread limit for {name}")
    readback["native_pools"] = pools
    if (not 1 <= len(readback["affinity"]) <= threads or not 1 <= threads <= 2
            or (readback["torch"], readback["interop"], readback["opencv"]) != (threads, 1, 1)
            or (utils.NUM_THREADS, torch_utils.NUM_THREADS) != (threads, threads)
            or any(not 1 <= n <= threads for n in pools.values())):
        raise RuntimeError(f"CPU worker limits differ from the protocol: {readback}")
    return readback
