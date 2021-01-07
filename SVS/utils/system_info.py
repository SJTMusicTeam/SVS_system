"""Copyright [2020] [Jiatong Shi].

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import sys
import torch

# import cupy
# import cupy.cuda
# from cupy.cuda import cudnn


def print_system_info():
    """print_system_info."""
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
