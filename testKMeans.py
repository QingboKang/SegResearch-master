import numpy as np
import cv2
import os

import evaluation

def doKMeans(imagepath, savepath):
    img = cv2.imread(imagepath)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS )

    center = np.uint8(center)
    center[0:3] = [0, 0, 0]
    center[3] = [255, 255, 255]

    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

    # noise removal
    kernel = np.ones((5, 5), np.uint8)

    gray = cv2.medianBlur(gray, 5)

    bin_img = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)

    cv2.imwrite(savepath, bin_img)
    ratio = evaluation.get_white_ratio(bin_img)
    area_count, area_mean, area_std = evaluation.count_contours_stddev(bin_img)

    print (savepath, " ", str(ratio), " ", str(area_count), " ", str(area_mean), " ", str(area_std))
    '''
    cv2.imshow('orig', gray)
    cv2.imshow('close', bin_img)
    cv2.waitKey(0)
    '''


orig_dir = '../Tissue images/'
dst_dir = '../kmeans_results/'

tif_files = os.listdir(orig_dir)
for file in tif_files:
    if not os.path.isdir(file):
        tiffilename = orig_dir + file
        resultFileName = dst_dir + file[0:file.find('.tif')] + '.png'
        doKMeans(tiffilename, resultFileName)
        #print (resultFileName)
