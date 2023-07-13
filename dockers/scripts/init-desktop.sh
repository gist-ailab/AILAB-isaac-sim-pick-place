#!/bin/bash
# Copyright (c) 2023 Ho Kim (ho.kim@ulagbulag.io). All rights reserved.
# Use of this source code is governed by a GPL-3-style license that can be
# found in the LICENSE file.

# Prehibit errors
set -e -o pipefail
# Verbose
set -x

# Skip if already initialized
LOCKFILE="${HOME}/.kiss-lock"
if [ -f "${LOCKFILE}" ]; then
    exec true
fi

# Install 3rd-party libraries
function download_library() {
    # Configure variables
    URL=$1
    filename="${URL##*/}"
    extension="${filename##*.}"

    # Download a file
    wget "${URL}" -o "${filename}"

    # Extract the file
    case "${extension}" in
    "tar")
        tar -xf "${filename}"
        ;;
    "zip")
        unzip -o -q "${filename}"
        ;;
    *)
        echo "Skipping extracting a file: ${filename}"
        ;;
    esac

    # Remove the original file
    rm -f "${filename}"
}

# ## Fonts
# FONT_HOME="${HOME}/.local/share/fonts"
# mkdir -p "${FONT_HOME}" && pushd "${FONT_HOME}"
#   for url in ${KISS_DESKTOP_FONTS_URL}; do
#     download_library "${url}"
#   done
#   fc-cache -f
# popd

# ## Icons
# ICON_HOME="${HOME}/.local/share/icons"
# mkdir -p "${ICON_HOME}" && pushd "${ICON_HOME}"
#   for url in ${KISS_DESKTOP_ICONS_URL}; do
#     download_library "${url}"
#   done
# popd

# ## Themes
# THEME_HOME="${HOME}/.local/share/themes"
# mkdir -p "${THEME_HOME}" && pushd "${THEME_HOME}"
#   for url in ${KISS_DESKTOP_THEMES_URL}; do
#     download_library "${url}"
#   done
# popd

# Cleanup

## Cleanup templates
rm -rf "${HOME}/.git" ${HOME}/*

## Cleanup ZSH configurations
rm -rf \
    "${HOME}/.oh-my-zsh" \
    "${HOME}/.zshrc" \
    "${HOME}/.zshrc.pre-oh-my-zsh"

# Download and install templates
pushd "${HOME}"
git init .
git remote add origin "${KISS_DESKTOP_TEMPLATE_GIT}"
# git fetch --all
git pull "origin" "${KISS_DESKTOP_TEMPLATE_GIT_BRANCH}"
popd

# Backup .zshrc
pushd "${HOME}"
mv .zshrc .zshrc-bak
popd

# ZSH Theme
pushd "${HOME}"
sh -c "$(curl -fsSL "https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh")"
git clone --depth=1 \
    "https://github.com/romkatv/powerlevel10k.git" \
    "${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k"
popd

# Restore .zshrc
pushd "${HOME}"
rm -f .zshrc
mv .zshrc-bak .zshrc
popd

# Finished!
exec touch "${LOCKFILE}"
