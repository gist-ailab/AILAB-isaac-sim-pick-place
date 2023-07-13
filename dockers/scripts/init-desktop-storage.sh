#!/bin/bash
# Copyright (c) 2023 Ho Kim (ho.kim@ulagbulag.io). All rights reserved.
# Use of this source code is governed by a GPL-3-style license that can be
# found in the LICENSE file.

# Prehibit errors
set -e -o pipefail
# Verbose
set -x

function _mount_overlayfs() {
    local src=$1
    local dst=$2
    local read_only="$3"

    if [[ "x${read_only}" != 'xtrue' ]]; then
        rm -rf "${dst}"
        ln -sf "${src}" "${dst}"
        return
    fi

    local src_name="$(basename "${src}")"

    local lowerdir="${src}"
    local upperdir="/tmp/${src_name}/upperdir"
    local workdir="/tmp/${src_name}/workdir"

    rm -rf "${upperdir}" "${workdir}"
    mkdir -p "${upperdir}" "${workdir}"

    fuse-overlayfs -o "auto_unmount,lowerdir=${lowerdir},upperdir=${upperdir},workdir=${workdir}" "${dst}"
}

_mount_overlayfs '/mnt/static' "/opt/public" 'true'
_mount_overlayfs '/mnt/public' "${HOME}/Public" 'false'
