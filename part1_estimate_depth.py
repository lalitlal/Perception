import os
import sys

import cv2
import numpy as np
import kitti_dataHandler
from matplotlib import pyplot as plt
# from PIL import Image

TRAINING = False
VIS = False
def main():

    ################
    # Options
    ################
    # Input dir and output dir
    if TRAINING:
        disp_dir = 'data/train/disparity'
        calib_dir = 'data/train/calib'
        left_img_dir = 'data/train/left'
        sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
    else:
        disp_dir = 'data/test/disparity'
        calib_dir = 'data/test/calib' 
        left_img_dir = 'data/test/left'
        est_depth_dir = 'data/test/est_depths/'
        sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in (sample_list):
        # Read disparity map
        disparity_image_path = disp_dir + '/' + sample_name + '.png'
        disp_map = cv2.imread(disparity_image_path, cv2.IMREAD_GRAYSCALE)

        # Read calibration info
        frame_calib_path = calib_dir + '/' + sample_name + '.txt'
        frame_calib = kitti_dataHandler.read_frame_calib(frame_calib_path)
        stereo_calib = kitti_dataHandler.get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Calculate depth (z = f*B/disp)
        zero_id = np.where(disp_map == 0)
        z = stereo_calib.f * stereo_calib.baseline / disp_map

        # Discard pixels past 80m and less than 10cm
        z[zero_id] = 0
        big_ind = np.where(z > 80)
        small_ind = np.where(z < 0.1)
        z[big_ind] = 0
        z[small_ind] = 0
        
        # Save depth map
        if VIS:
            plt.figure()
            left_img_path = left_img_dir + '/' + sample_name + '.png'
            left_img = cv2.imread(left_img_path)
            plt.imshow(left_img, cmap='gray')
            plt.imshow(z, cmap='jet', alpha=0.5)
            plt.show()
        
        if not TRAINING:
            depthMap = np.array(z)
            cv2.imwrite(est_depth_dir + sample_name + '.png', depthMap)
        
if __name__ == '__main__':
    main()
