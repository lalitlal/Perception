import os
import sys

import cv2
import numpy as np
import kitti_dataHandler
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix


TRAINING = False
VIS = False
EPSILON = 1 # 1m radius

def main():

    ################
    # Options
    ################
    if TRAINING:
        # Input dir and output dir
        left_img_dir = 'data/train/left'
        depth_dir = 'data/train/gt_depth'
        label_dir = 'data/train/gt_labels'
        gt_seg_dir = 'data/train/gt_segmentation'
        sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']

    else:
        # Input dir and output dir
        left_img_dir = 'data/test/left'
        # depth_dir = 'data/test/gt_depth' -- in original, but NOT USED, why is this here?
        # label_dir = 'data/test/gt_labels' -- in origin, but NOT USED, why is this here? 
        depth_dir = 'data/test/est_depths'
        label_dir = 'data/test/est_labels'
        output_seg_dir = 'data/test/est_segmentation'
        sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    final_pre = []
    final_rec = []
    for sample_name in sample_list:
    	# Read depth map
        depth_img_path = depth_dir + '/' + sample_name + '.png'
        gt_depth_map = cv2.imread(depth_img_path, cv2.IMREAD_GRAYSCALE)

        # Discard depths less than 10cm from the camera
        small_ind = np.where(gt_depth_map < 0.1)
        gt_depth_map[small_ind] = 0

        # Read 2d bbox
        # form is x1,y1,x2,y2
        bboxs = []
        lbl = kitti_dataHandler.read_labels(label_dir, sample_name)
        for obj in lbl:
            if obj.type.lower() == 'car':
                bboxs.append((obj.x1, obj.y1, obj.x2, obj.y2))

         # first create a clean slate mask
        cpy = gt_depth_map.copy()
        cpy[:,:] = 255
        x_max = np.shape(gt_depth_map)[1]
        y_max = np.shape(gt_depth_map)[0]
        # For each bbox
        for i, bbox in enumerate(bboxs):
            # Estimate the average depth of the objects
            x1,y1,x2,y2 = bbox
            w = x2-x1
            h = y2-y1
            y1_half = y1+(h/4) # actually the top of the ROI -> shrink it because it's more sparse
            y2_half = y2-(h/8) # actually the bottom -> bring this up a bit too to remove the ground readings

            cropped = np.asarray(gt_depth_map[int(y1):int(y2), int(x1):int(x2)])
            mean_ROI = np.asarray(gt_depth_map[int(y1_half):int(y2_half), int(x1):int(x2)]) # only use bottom half, it's more reliable
            min_samples = h*w * (2/4) #half of the image, else we'll just get the whole mean
            
            # first try to use the bottom half only
            if len(mean_ROI[mean_ROI!=0]) > min_samples:
                mean_est_depth = mean_ROI[mean_ROI != 0].mean()
            
            # if the bottom half didn't give us enough samples, try to use the whole ROI
            elif len(cropped[cropped!=0] > min_samples):
                mean_est_depth = cropped[cropped!=0].mean()
            
            # worst case, this object is out of the LIDAR range, assign the estimated depth to 0
            else:
                mean_est_depth = 0
    
            # Find the pixels within a certain distance from the centroid
            if mean_est_depth != 0:
                # find which regions are generally appropriate
                minthresh = mean_est_depth-EPSILON if mean_est_depth-EPSILON > 0 else mean_est_depth
                maxthresh = mean_est_depth+EPSILON
                
                _, mask = cv2.threshold(cropped, maxthresh, 255, cv2.THRESH_BINARY)
                
                # assignment of mask onto the final mask
                mask[mask > 0] = 255
                mask[ mask == 0] = 0 # assignment asks this to be 0 for all cars
                cpy[int(y1):int(y2),int(x1):int(x2)] = mask

            else:
                cpy[int(y1):int(y2),int(x1):int(x2)] = [0]

        if VIS:       
            plt.imshow(cpy, 'gray')
            plt.show()

        if TRAINING:
            # Evaluate Precision and Recall:
            # First read the gt mask:
            gt_seg_path = gt_seg_dir + '/' + sample_name + '.png'
            gt_seg_map = cv2.imread(gt_seg_path, cv2.IMREAD_GRAYSCALE)
            pred = cpy.copy()
            test = gt_seg_map.copy()
            test[test < 255] = 0

            # https://www.jeremyjordan.me/evaluating-image-segmentation-models/
            TP, FP, FN = 0, 0, 0
            h , w = np.shape(pred)
            # loop over the image, pixel by pixel
            for y in range(0, h):
                for x in range(0, w):
                    if test[y, x] != 255 and pred[y, x] == test[y,x]: 
                        TP += 1
                    elif test[y, x] != 255 and pred[y, x] != test[y,x]:
                        FN += 1
                    elif test[y, x] != pred[y,x] and pred[y, x] != 255:
                        FP += 1
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            print(np.mean(precision), np.mean(recall))
            final_pre.append(np.mean(precision))
            final_rec.append(np.mean(recall))

        # Save the segmentation mask
        if not TRAINING:
            estSegMap = np.array(cpy)
            cv2.imwrite(output_seg_dir + '/' + sample_name + '.png', estSegMap)
        
    if TRAINING:
        print(np.mean(final_pre), np.mean(final_rec))



if __name__ == '__main__':
    main()


# FINAL QUESTIONS:
# 1. Assignment says to submit them all as class 0 for car, rest = 255, but GT segmentation shows labelled instances (0, 1, 2)
