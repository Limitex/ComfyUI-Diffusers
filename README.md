# ComfyUI-Diffusers

This repository is a custom node in ComfyUI.

## Overview

This is a program that allows you to use Huggingface Diffusers module with ComfyUI. Additionally, Stream Diffusion is also available.

![Workflow](img/workflow.png)

In addition, real-time generation is possible by doing the following.
When running, please enable Auto Queue in Extra options.

![Workflow2](img/workflow2.png)

## Usage

Run the following command inside ComfyUI/custom_nodes.

```cmd
git clone https://github.com/Limitex/ComfyUI-Diffusers.git
cd ComfyUI-Diffusers
pip install -r requirements.txt
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
python -m streamdiffusion.tools.install-tensorrt
```

## Node

### Diffusers Pipeline Loader (DiffusersPipelineLoader)

### Diffusers Vae Loader (DiffusersVaeLoader)

### Diffusers Scheduler Loader (DiffusersSchedulerLoader)

### Diffusers Model Makeup (DiffusersModelMakeup)

### Diffusers Clip Text Encode (DiffusersClipTextEncode)

### Diffusers Sampler (DiffusersSampler)

### Diffusers Save Image (DiffusersSaveImage)

### StreamDiffusion Create Stream (StreamDiffusionCreateStream)

### StreamDiffusion Sampler (StreamDiffusionSampler)

### StreamDiffusion Warmup (StreamDiffusionWarmup)

### StreamDiffusion Fast Sampler (StreamDiffusionFastSampler)
