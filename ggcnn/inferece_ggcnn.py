import torch
import numpy as np
import cv2
from ggcnn.models.ggcnn2 import GGCNN2
from ggcnn.models.ggcnn import GGCNN
from ggcnn.models.common import post_process_output
from ggcnn.utils.dataset_processing import evaluation, grasp

def inference_ggcnn(depth, bbox):
    cy, cx = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)
    crop_range = 150
    cx = 1080-crop_range
    
    cropped_depth = depth[cx-crop_range: cx+crop_range, cy-crop_range: cy+crop_range]
    print(cx, cy)
    print(cropped_depth.shape)
    # cropped_depth = np.transpose(cropped_depth, (2, 0, 1))
    # cropped_depth = cropped_depth[0]
    cropped_depth = np.clip((cropped_depth - cropped_depth.mean()), -1, 1)
    depthT = torch.from_numpy(cropped_depth.reshape(1, 1, 2*crop_range, 2*crop_range).astype(np.float32)).cuda()
    
    ## GGCNN Network
    net = GGCNN()
    net.load_state_dict(torch.load("/home/nam/.local/share/ov/pkg/isaac_sim-2022.2.0/workspace/ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt"))
    net.cuda()
    
    with torch.no_grad():
        pred_out = net(depthT)
    q_img, ang_img, width_img = post_process_output(pred_out[0], pred_out[1], pred_out[2], pred_out[3])
        
    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
    
    if len(grasps):
        angle = grasps[0].angle
        length = grasps[0].length
        width = grasps[0].width
        center = grasps[0].center
        
    center = [cx + center[0] - crop_range, cy + center[1] - crop_range]
    
    return angle, length, width, center
        
