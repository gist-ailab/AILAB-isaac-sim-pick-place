# Add symlink
ln -s exts/omni.isaac.examples/omni/isaac/examples /isaac-sim/extension_examples

# libcuda
ln -sf /usr/lib64/libcuda.so.1 /usr/lib64/libcuda.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64