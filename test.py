import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


def CountContoursStddev(binImg):
    colorImg = cv2.cvtColor(binImg, cv2.COLOR_GRAY2BGR)
    im2, cnts, hierchy = cv2.findContours( binImg.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE )

    lstAreas = []
    area_count = 0;
    for c in cnts:
        area = cv2.contourArea(c)
        if area < binImg.size * 0.001:
            area_count += 1
            lstAreas.append(area)
            color = list(np.random.random(size=3) * 256)
            cv2.drawContours(colorImg, [c], -1, color, 2)

   # cv2.imshow("contours", colorImg)
   # cv2.waitKey(0)
    arr = np.array(lstAreas)
    area_std = np.std(arr)
    area_mean = np.mean(arr)
    return area_count, area_std, area_mean

    '''
    removal_size: 3
    open_iterations: 3
    dilate_iterations: 3
    fThreshold: 0.25
    '''
def ImageWatershed(gray, dstImageName, dstMarkName, blocksize, constant, removal_size, open_iterations, \
                   dilate_iterations, fThreshold):
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blocksize, constant)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.25*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    bin_mask = np.ones(markers.shape) * 255
    bin_mask[markers == 1] = 0
    bin_mask = bin_mask.astype(np.uint8)

    count_holes, std_holes, mean_holes = CountContoursStddev(bin_mask)

    #cv2.imshow('binary', thresh);
    return count_holes, std_holes, mean_holes, bin_mask


img = cv2.imread('../bad/29.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

block_size = range(11, 501, 50)
constants = range(-20, 20, 10)
removal_size = range(3, 11, 2)
open_iterations = range(1, 5, 1)
dilate_iterations = range(1, 5, 1)
fThreshold = np.arange(0.1, 0.5, 0.02)

import sys
import math
optimal_result = -1000

lstCounts = []
lstMeans = []
for size in block_size:
    for const in constants:
        for removalSize in removal_size:
            for open_times in open_iterations:
                for dilate_times in dilate_iterations:
                    for thres in fThreshold:
                        count_holes, std_holes, mean_holes, bin_mask = ImageWatershed(gray, "", "", size, const, \
                                    removalSize, open_times, dilate_times, thres)
                        lstCounts.append(count_holes)
                        lstMeans.append(mean_holes)



lstCounts = [round(i / 10) for i in lstCounts]
lstCounts = sorted(lstCounts)
tempset = set(lstCounts)
for item in tempset:
    print("%s: %s" % (item, lstCounts.count(item)))

lstMeans = [round(i / 10) for i in lstMeans]
lstMeans = sorted(lstMeans)
tempset = set(lstMeans)
for item in tempset:
    print ("%s: %s" % (item, lstMeans.count(item)))
#cv2.waitKey(0)

