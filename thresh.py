import numpy as np
import cv2
import os
import normalizeStaining

def doBinaryThreshold( blocksize, constant, gray, imagesavepath ):
    for const in range(-30, 30, 2):
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blocksize,
                                           const)
        ratio = np.count_nonzero(thresh) / thresh.size
        if ratio <= 0.3 and ratio >= 0.15:
            imagesavepath = imagesavepath + str(const) + "_" + str(ratio) + ".png"
            cv2.imwrite(imagesavepath, thresh)
            break

    return


block_size = range(11, 801, 10)
constants = range(0, 5, 5)

img_dir = '../Tissue images/'
result_dir = '../thres_results/'

tif_files = os.listdir(img_dir)
for file in tif_files:
    if not os.path.isdir(file):
        tiffilename = img_dir + file
        sub_result_dir = result_dir + file[0:file.find('.tif')] + '/'
        if not os.path.exists(sub_result_dir):
            os.mkdir(sub_result_dir)

        # rgb --> gray
        img = cv2.imread(tiffilename)
        '''Color Normalization'''
        img = img.astype('float32')
        img = normalizeStaining.normalizeStaining(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for blksize in block_size:
            for const in constants:
                binaryimagepath = sub_result_dir + str(blksize) + '_'
                doBinaryThreshold(blksize, const, gray, binaryimagepath)

        print (tiffilename)
        print (sub_result_dir)