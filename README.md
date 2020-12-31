# SVS_system
An open-source system that works on singing voice synthesis

## Environment Install

### Steps1: Git clone SVS_system

```
$ cd <any-place>
$ git clone https://github.com/SJTMusicTeam/SVS_system.git
```

### Step2 [Optional]: Put complied Kaldi under SVS_system/tools

If you have complied Kaldi at Step1, put it under `tools`.

```
$ cd <SVS_system-root>/tools
$ ln -s <kaldi-root> .
```

If you do not have `SVS_system/tools/kaldi`, when `make`, Kaldi repository is automatcally put without compliling (For now, we only use some basic shell in Kaldi, so no need for complied versions).

### Step3: Setup Python environment

You have to create `<SVS_system-root>/tools/activate_python.sh` to specify the Python interpreter used in SVS_system recipes. (To understand how SVS_system specifies Python, see path.sh for example.)

We also have some scripts to generate `tools/activate_python.sh`.

* Option A: setup anaconda environment

```
$ cd <SVS_system-root>/tools
$ ./setup_anaconda.sh [output-dir-name|default=venv] [conda-env-name|default=root] [python-version|default=none]
# e.g.
$ ./setup_anaconda.sh anaconda svs 3.8
```

This script tries to create a new miniconda if the output directory doesn’t exist. If you already have Anaconda and you’ll use it then,

```
$ cd <SVS_system-root>/tools
$ CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..
$ ./setup_anaconda.sh ${CONDA_TOOLS_DIR} [conda-env-name] [python-version]
# e.g.
$ ./setup_anaconda.sh ${CONDA_TOOLS_DIR} svs 3.8
```

* Option B: Setup system Python environment
```
$ cd <SVS_system-root>/tools
$ ./setup_python.sh $(command -v python3)
```

* Option C: Without setting Python environment

`Option B` and `Option C` are almost same. This option might be suitable for Google colab.

```
$ cd <SVS_system-root>/tools
$ rm -f activate_python.sh && touch activate_python.sh
```
### Step4: Install SVS
```
$ cd <SVS_system-root>/tools
$ make
```

The Makefile tries to install ESPnet and all dependencies including PyTorch. You can also specify PyTorch version, for example:
```
$ cd <SVS_system-root>/tools
$ make TH_VERSION=1.3.1
```


Note that the CUDA version is derived from `nvcc` command. If you’d like to specify the other CUDA version, you need to give `CUDA_VERSION`.
```
$ cd <SVS_system-root>/tools
$ make TH_VERSION=1.3.1 CUDA_VERSION=10.1
```

If you don’t have `nvcc` command, packages are installed for CPU mode by default. If you’ll turn it on manually, give `CPU_ONLY` option.
```
$ cd <SVS_system-root>/tools
$ make CPU_ONLY=0
```

### Step5: Check installation

You can check whether your installation is successfully finished by
```
$ cd <SVS_system-root>/tools
$ . ./activate_python.sh; python3 check_install.py
```

Note that this check is always called in the last stage of the above installation.

## Running Instruction

For example: \
    `cd egs/public_dataset/kiritan`  \
    `./train.sh` 

- For CLSP User, using clsp_wrapper to use qsub.
- For other user, using train.sh or infer.sh to run.

Please refer to configuration file (e.g. train.yaml) for parameters.


