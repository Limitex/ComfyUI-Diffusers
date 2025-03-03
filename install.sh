#!/bin/bash
source ../../venv/bin/activate

pip install -r requirements.txt
python python -m streamdiffusion.tools.install-tensorrt

git clone https://github.com/cumulo-autumn/StreamDiffusion.git
python -m streamdiffusion.tools.install-tensorrt

if [[ $(uname) == "Darwin" ]]; then # for MacOS / MPS
    cd StreamDiffusion
    git apply ../ignore_cuda_for_mac.patch
    pip install .
else
    python -m streamdiffusion.tools.install-tensorrt # for PC / CUDA
fi
