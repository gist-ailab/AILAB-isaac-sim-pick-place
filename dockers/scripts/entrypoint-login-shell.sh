#!/bin/bash
# Copyright (c) 2023 Ho Kim (ho.kim@ulagbulag.io). All rights reserved.
# Use of this source code is governed by a GPL-3-style license that can be
# found in the LICENSE file.

# Catch trap signals
trap "echo 'Gracefully terminating...'; exit" INT TERM
trap "echo 'Terminated.'; exit" EXIT

# Get the screen size
SCREEN_WIDTH="640"
SCREEN_HEIGHT="480"

# Configure screen size
function update_screen_size() {
    echo "Finding displays..."
    screens="$(xrandr --current | grep ' connected ' | awk '{print $1}')"
    if [ "x${screens}" == "x" ]; then
        echo 'Display not found!'
        exit 1
    fi

    for screen in $(echo -en "${screens}"); do
        echo "Fixing screen size (${screen})..."
        until [ "$(
            xrandr --current |
                grep ' connected' |
                grep -Po '[0-9]+x[0-9]+' |
                head -n1
        )" == "${SCREEN_WIDTH}x${SCREEN_HEIGHT}" ]; do
            xrandr --output "${screen}" --mode "${SCREEN_WIDTH}x${SCREEN_HEIGHT}"
            sleep 1
        done
    done

    # Disable screen blanking
    echo "Disabling screen blanking..."
    xset -dpms
    xset s off
}

# Configure firefox window
function update_window() {
    classname="$1"

    xdotool search --classname "${classname}" set_window --name 'Welcome'
    xdotool search --classname "${classname}" windowsize "${SCREEN_WIDTH}" "${SCREEN_HEIGHT}"
    xdotool search --classname "${classname}" windowfocus
    update_screen_size
}

while :; do
    echo "Waiting until logged out..."
    while [ "$(
        kubectl get node "${NODENAME}" \
            --output jsonpath \
            --template '{.metadata.labels.ark\.ulagbulag\.io/bind}'
    )" == 'true' ]; do
        sleep 3
    done

    update_screen_size

    echo "Executing a login shell..."
    firefox \
        --first-startup \
        --private \
        --window-size "${SCREEN_WIDTH},${SCREEN_HEIGHT}" \
        --kiosk "${VINE_BASTION_ENTRYPOINT}/box/${NODENAME}/login" &
    PID=$!
    TIMESTAMP=$(date -u +%s)

    echo "Waiting until window is ready..."
    sleep 1
    until xdotool search --classname 'Navigator' >/dev/null; do
        sleep 0.5
    done

    echo "Resizing window to fullscreen..."
    update_window 'Navigator'

    echo "Waiting until login is succeeded..."
    until [ "$(
        kubectl get node "${NODENAME}" \
            --output jsonpath \
            --template '{.metadata.labels.ark\.ulagbulag\.io/bind}'
    )" == 'true' ]; do
        # Session Timeout
        NOW=$(date -u +%s)
        TIMEOUT_SECS="300" # 5 minutes
        if ((NOW - TIMESTAMP > TIMEOUT_SECS)); then
            echo "Session timeout ($(date))"
            break
        fi

        sleep 1
    done

    echo "Stopping firefox..."
    kill "${PID}" 2>/dev/null || true
done
