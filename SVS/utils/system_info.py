import sys
import torch

# import cupy
# import cupy.cuda
# from cupy.cuda import cudnn


def print_system_info():
    pyver = sys.version.replace("\n", " ")
    print(f"python version: {pyver}")
    print(f"pytorch version: {torch.__version__}")
    # print(f"cupy version: {cupy.__version__}")

    if torch.cuda.is_available():
        # print(f"cuda version: {cupy.cuda.runtime.runtimeGetVersion()}")
        print(f"cuda version: {torch.version.cuda}")

        # print(f"cudnn version: {cudnn.getVersion()}")
        print(f"cudnn version: {torch.backends.cudnn.version()}")


# print_system_info()
