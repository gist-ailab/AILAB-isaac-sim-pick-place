# Isaac Sim Dockerfiles

## Introduction

This repository contains Dockerfiles for building an Isaac Sim container.

### Pre-Requisites

Before getting started, ensure that the system has the latest [NVIDIA Driver](https://www.nvidia.com/en-us/drivers/unix/) and the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) installed.


## Build

Clone this repository and then build the image:

```bash
docker login nvcr.io

id : $oauthtoken
pwd : Njc5dHR0b2QwZTh0dTFtNW5ydXI4Y3JtNm46MGVkM2VjODctZTk1Ni00NmNjLTkxNDEtYTdmMjNlNjllMjNj 
```

```bash
docker build --pull -t \
  isaac-sim:2022.2.1-ubuntu20.04 \
  --build-arg ISAACSIM_VERSION=2022.2.1 \
  --build-arg BASE_DIST=ubuntu20.04 \
  --build-arg CUDA_VERSION=11.4.2 \
  --build-arg VULKAN_SDK_VERSION=1.3.224.1 \
  --file Dockerfile.2022.2.1-ubuntu20.04 .
```

## Usage

To run the container and start Isaac Sim as a windowed app:

```bash
xhost +
docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -e DISPLAY \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  isaac-sim:2022.2.1-ubuntu20.04
```
