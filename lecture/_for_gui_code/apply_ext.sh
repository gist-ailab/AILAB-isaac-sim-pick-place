#! /usr/bin/sh

# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# ailab extension apply bash
# ---- ---- ---- ----

{
cp -Rv ailab_examples /isaac-sim/exts/omni.isaac.examples/omni/isaac/examples/ \
&&
cp -Rv ailab_script /isaac-sim/exts/omni.isaac.examples/omni/isaac/examples/ \
&&
cp -v extension.toml /isaac-sim/exts/omni.isaac.examples/config/ \
&&
echo "\n\n.. apply \'isaac-sim ailab ext\' complete\n"
} ||
{
echo "\n\n!! apply \'isaac-sim ailab ext\' fail !!\n"
}