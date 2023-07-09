import glob
import shutil

### ycb usd file path
a = glob.glob("/home/ailab/Workspace/minhwan/ycb/*/*/*.png")
### ycb urdf file path
b = glob.glob("/home/ailab/Workspace/minhwan/YCB/*/*/texture_map.png")
a.sort()
b.sort()

for i in range(len(a)):
    shutil.copy(b[i],a[i])