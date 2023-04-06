import torch
import numpy as np
# import cv2
# from ggcnn.models.ggcnn2 import GGCNN2
from ggcnn.models.ggcnn import GGCNN
from ggcnn.models.common import post_process_output
from ggcnn.utils.dataset_processing import grasp        #, evaluation
# import matplotlib.pyplot as plt


def inference_ggcnn(rgb, depth, mask, bbox, crop_range=200):
    # rgb = cv2.imread("/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/rgbcam_3_45_rgb.png")
    # depth = cv2.imread("/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/rgbcam_3_45_newd.png")
    # depth = depth / 255
    # mask = cv2.imread("/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/rgbcam_3_45_mask.png")
    ## height, width = 1080, 1920 = y, x

    print('depth shape', depth.shape)

    # # bbox = np.array([897, 779, 991, 890])   # rgbcam
    # # bbox = np.array([918, 701, 980, 775])   # dcam
    # bbox = np.array([662, 944, 767, 1064])

    cy, cx = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)
    # crop_range = 300
    cx = 1080-crop_range
    print(f'center  x={cx}, y={cy}')

    print(cx-crop_range, cx+crop_range, cy-crop_range, cy+crop_range)
    cropped_rgb = rgb[cx-crop_range: cx+crop_range, cy-crop_range: cy+crop_range]
    cropped_depth = depth[cx-crop_range: cx+crop_range, cy-crop_range: cy+crop_range]

    cropped_depth = np.transpose(cropped_depth, (2, 0, 1))
    cropped_depth = cropped_depth[0]
    print('cropped_depth shape', cropped_depth.shape)
    cropped_depth = np.clip((cropped_depth - cropped_depth.mean()), -1, 1)
    depthT = torch.from_numpy(cropped_depth.reshape(1, 1, 2*crop_range, 2*crop_range).astype(np.float32)).cuda()

    ## GGCNN Network
    net = GGCNN()
    net.load_state_dict(torch.load("/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt"))
    net.cuda()

    with torch.no_grad():
        pred_out = net(depthT)
    print(torch.unique(pred_out[3]))
    q_img, ang_img, width_img = post_process_output(pred_out[0], pred_out[1], pred_out[2], pred_out[3])
    print(width_img[134, 233])


    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
    print('number of grasps', len(grasps))
    if len(grasps):
        # if vis == True:
        # ray = self.cam_for_ggcnn.pixel_to_xyz()
        # center = ray[grasps[0].center]
        angle = grasps[0].angle
        length = grasps[0].length
        width = grasps[0].width
        print(angle, length, width)
        print('grasp center', grasps[0].center)

    # evaluation.plot_output(rgb, depth, q_img, ang_img, no_grasps=1)
    # evaluation.plot_output(cropped_rgb, cropped_depth, q_img, ang_img, no_grasps=1)

    return grasps

