# Docker Pull Container
```bash
docker pull --tls-verify=false registry.ark.svc.ops.openark/library/isaac-sim:2022.2.1-ubuntu22.04_v3
```

# Run container
```bash
podman run -it --entrypoint bash --name isaac-sim --device nvidia.com/gpu=all -e "ACCEPT_EULA=Y" --rm --network=host \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY \
-v /home/user/Desktop:/home/workspace \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
registry.ark.svc.ops.openark/library/isaac-sim:2022.2.1-ubuntu22.04_v3
```

# 컨테이너 빠져나오기 (background로 실행)
- ctrl + p + q


# VSCODE 설정
- dev container 확장 설치
- docker container attach
- open project folder