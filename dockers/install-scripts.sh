#!/bin/bash
# Copyright (c) 2023 Ho Kim (ho.kim@ulagbulag.io). All rights reserved.
# Use of this source code is governed by a GPL-3-style license that can be
# found in the LICENSE file.

# Prehibit errors
set -e -o pipefail
# Verbose
set -x

###########################################################
#   Install VINE Desktop Scripts                          #
###########################################################

echo "- Installing VINE Desktop scripts ... "

kubectl create configmap "vine-desktop-scripts" \
    --namespace=vine-guest \
    --from-file="./scripts" \
    --output=yaml \
    --dry-run=client |
    kubectl apply -f -

# Finished!
echo "Installed!"
