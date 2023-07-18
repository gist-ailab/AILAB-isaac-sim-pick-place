docker build --pull -t \
  registry.ark.svc.ops.openark/library/isaac-sim:2022.2.1-ubuntu22.04_v3 \
  --build-arg ISAACSIM_VERSION=2022.2.1 \
  --build-arg BASE_DIST=ubuntu20.04 \
  --build-arg CUDA_VERSION=11.4.2 \
  --build-arg VULKAN_SDK_VERSION=1.3.224.1 \
  --file Dockerfile.2022.2.1-ubuntu22.04 .


podman run -it --entrypoint bash --name isaac-sim --device nvidia.com/gpu=all -e "ACCEPT_EULA=Y" --rm --network=host \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY \
-v /home:/home \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
registry.ark.svc.ops.openark/library/isaac-sim:2022.2.1-ubuntu22.04_v3