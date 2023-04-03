import numpy as np
import cv2
import matplotlib.pyplot as plt



def depth_image_from_distance_image(distance, intrinsics):
    """Computes depth image from distance image.
    
    Background pixels have depth of 0
    
    Args:
        distance: HxW float array (meters)
        intrinsics: 3x3 float array
    
    Returns:
        z: HxW float array (meters)
    
    """
    fx = intrinsics[0][0]
    cx = intrinsics[0][2]
    fy = intrinsics[1][1]
    cy = intrinsics[1][2]
    
    height, width = distance.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    
    x_over_z = (px - cx) / fx
    y_over_z = (py - cy) / fy
    
    z = distance / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    return z




depth_path = "/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/rgbcam_3_45_d.png"
new_depth_path = "/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/rgbcam_3_45_newd.png"
depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
# cv2.resize(depth, (1920, 1080))
print(type(depth))
print(depth.size)   # 1920*1080*3       6220800 # 1920*1080*1       2073600
print(depth.shape)  # (1080, 1920)
# intrinsics = [[940.50757, 0.0, 960.0], [0.0, 529.0355, 540.0], [0.0, 0.0, 1.0]]         # depth camera intrinsics
intrinsics = [[1398.3395, 0.0, 960.0], [0.0, 786.56598, 540.0], [0.0, 0.0, 1.0]]        # rgb camera intrinsics



new_depth = np.uint16(depth_image_from_distance_image(depth, intrinsics))
plt.imshow(depth)
plt.show()

plt.imshow(new_depth)
plt.show()
cv2.imwrite(new_depth_path, new_depth*255)