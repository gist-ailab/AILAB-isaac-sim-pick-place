# ailab extension apply bash

{
cp -Rv ./AILAB-isaac-sim-pick-place/_for_gui_code/ailab_examples ./exts/omni.isaac.examples/omni/isaac/examples/ \
&&
cp -Rv ./AILAB-isaac-sim-pick-place/_for_gui_code/ailab_script ./exts/omni.isaac.examples/omni/isaac/examples/ \
&&
cp -v ./AILAB-isaac-sim-pick-place/_for_gui_code/extension.toml ./exts/omni.isaac.examples/config/ \
&&
echo "\n\n.. apply complete\n"
} ||
{
echo "\n\n!! apply fail !!\n"
}
