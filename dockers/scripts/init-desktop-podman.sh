#!/bin/bash
# Copyright (c) 2023 Ho Kim (ho.kim@ulagbulag.io). All rights reserved.
# Use of this source code is governed by a GPL-3-style license that can be
# found in the LICENSE file.

# Prehibit errors
set -e -o pipefail
# Verbose
set -x

# Copy podman containers configuration file
mkdir -p "${HOME}/.config/containers"
rm -rf "${HOME}/.config/containers/containers.conf"
cp /etc/containers/podman-containers.conf "${HOME}/.config/containers/containers.conf"

# Initialize rootless podman
podman system migrate

# Generate a CDI specification that refers to all NVIDIA devices
nvidia-ctk cdi generate --device-name-strategy=type-index --format=json > /etc/cdi/nvidia.json
