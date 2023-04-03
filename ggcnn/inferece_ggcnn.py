import torch
import numpy as np
import cv2
from models.ggcnn2 import GGCNN2
from models.ggcnn import GGCNN
from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
import matplotlib.pyplot as plt


# def grasp_ggcnn(self, vis=False):
    # Grasp with GGCNN
# depth size : (200, 200)
# rgb, depth, _ = self.cam_for_ggcnn.shot()
# print(np.unique(depth))
# depth = (1 - depth) * 700
# cv2.imwrite("output/depth_for_ggcnn.png", depth)
rgb = cv2.imread("/home/nam/.local/share/ov/pkg/isaac_sim-2022.2.0/workspace/data/rgb_image_3.png")
depth = cv2.imread("/home/nam/.local/share/ov/pkg/isaac_sim-2022.2.0/workspace/data/depth_image_3.png")
depth = depth / 255
mask = cv2.imread("/home/nam/.local/share/ov/pkg/isaac_sim-2022.2.0/workspace/data/mask_3.png")
# print(np.unique(mask))
print(np.unique(depth))

# depthplot = plt.imshow(depth)
# plt.show()
# maskplot = plt.imshow(mask*100)
# plt.show()
# bbox = np.array([916, 706, 983, 777]) # scale 0.3
# bbox = np.array([864, 660, 1028, 841]) # scale 0.7
# bbox = np.array([830, 709, 1066, 975]) # scale 0.7, height 0.3
bbox = np.array([897, 779, 991, 890]) # scale 0.3, height 0.3, depth *100

center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
left = max(0, min(int(center[1]) - 200 // 2, 640 - 200))
top = max(0, min(int(center[0]) - 200 // 2, 480 - 200))
# cropped_rgb = rgb[bbox[1]-200: bbox[3]+200, bbox[0]-200: bbox[2]+200]
# cropped_depth = depth[bbox[1]-200: bbox[3]+200, bbox[0]-200: bbox[2]+200]
# print(int(center[0]))
cropped_rgb = rgb[int(center[1])-200: int(center[1])+200, int(center[0])-200: int(center[0])+200]
cropped_depth = depth[int(center[1])-200: int(center[1])+200, int(center[0])-200: int(center[0])+200]
# cropped_rgb = rgb
# cropped_depth = depth
# rgb = np.expand_dims(rgb, 0)
# depth = np.expand_dims(depth, 0)
# print(depth.shape)
# depth = np.transpose(depth, (2, 0, 1))
# depth = depth[0]
# depth = np.clip((depth - depth.mean()), -1, 1)
cropped_depth = np.transpose(cropped_depth, (2, 0, 1))
cropped_depth = cropped_depth[0]
cropped_depth = np.clip((cropped_depth - cropped_depth.mean()), -1, 1)
print(np.unique(cropped_depth))
# print(cropped_depth.shape)
depthT = torch.from_numpy(cropped_depth.reshape(1, 1, 400, 400).astype(np.float32)).cuda()
print(torch.unique(depthT))
net = GGCNN()
net.load_state_dict(torch.load("/home/nam/.local/share/ov/pkg/isaac_sim-2022.2.0/workspace/ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt"))
net.cuda()

with torch.no_grad():
    pred_out = net(depthT)
print(torch.unique(pred_out[3]))
q_img, ang_img, width_img = post_process_output(pred_out[0], pred_out[1], pred_out[2], pred_out[3])
print(width_img[134, 233])
grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
# if vis == True:
# ray = self.cam_for_ggcnn.pixel_to_xyz()
# center = ray[grasps[0].center]
angle = grasps[0].angle
length = grasps[0].length
width = grasps[0].width

print(angle, length, width)
print(grasps[0].center)

# evaluation.plot_output(rgb, depth, q_img, ang_img, no_grasps=1)
evaluation.plot_output(cropped_rgb, cropped_depth, q_img, ang_img, no_grasps=1)