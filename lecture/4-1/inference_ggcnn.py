import torch
import numpy as np
import cv2
import os
from ggcnn.models.ggcnn2 import GGCNN2
from ggcnn.models.ggcnn import GGCNN
from ggcnn.models.common import post_process_output
from ggcnn.utils.dataset_processing import grasp, evaluation
import matplotlib.pyplot as plt


def inference_ggcnn(rgb, depth, bbox, crop_range=200):

    cy, cx = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)
    cropped_rgb = rgb[cx-crop_range: cx+crop_range, cy-crop_range: cy+crop_range]
    cropped_depth = depth[cx-crop_range: cx+crop_range, cy-crop_range: cy+crop_range]

    # cropped_depth = np.transpose(cropped_depth, (2, 0, 1))
    # cropped_depth = cropped_depth[0]
    cropped_depth = np.clip((cropped_depth - cropped_depth.mean()), -1, 1)
    depthT = torch.from_numpy(cropped_depth.reshape(1, 2*crop_range, 2*crop_range).astype(np.float32)).cuda()

    ## GGCNN Network
    net = GGCNN()
    dir = os.path.dirname(os.path.realpath(__file__))
    path = "/ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt"
    path = dir + path
    net.load_state_dict(torch.load(path))
    net.cuda()

    with torch.no_grad():
        pred_out = net(depthT)
    q_img, ang_img, width_img = post_process_output(pred_out[0], pred_out[1], pred_out[2], pred_out[3])


    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
    print('number of grasps', len(grasps))
    if len(grasps):
        center = grasps[0].center
        angle = grasps[0].angle
        length = grasps[0].length
        width = grasps[0].width
        
    center = (cy-crop_range+center[1], cx-crop_range+center[0])

    evaluation.plot_output(cropped_rgb, cropped_depth, q_img, ang_img, no_grasps=1)

    return angle, length, width, center

