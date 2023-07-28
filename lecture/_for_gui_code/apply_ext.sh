#! /usr/bin/sh

# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# ailab extension apply bash
# ---- ---- ---- ----

{
cp -Rv ailab_examples /home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/exts/omni.isaac.examples/omni/isaac/examples/ \
&&
cp -Rv ailab_script /home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/exts/omni.isaac.examples/omni/isaac/examples/ \
&&
cp -v extension.toml /home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/exts/omni.isaac.examples/config/ \
&&
echo "\n\n.. apply \'/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0 ailab ext\' complete\n"
} ||
{
echo "\n\n!! apply \'/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0 ailab ext\' fail !!\n"
}
