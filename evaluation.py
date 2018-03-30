import cv2
import numpy as np


def get_white_ratio(binImg):

    ratio = np.count_nonzero(binImg) / binImg.size
    return ratio


def count_contours_stddev(binImg):
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
    area_mean = np.mean(arr)
    area_std = np.std(arr)
    return area_count, area_mean, area_std