import numpy as np
import cv2
import os

maskimgsdir = "../Mask/"
origimgsdir = "../Tissue images/"

lstMaskRatio = []
lstOTSURatio = []

for i in range(1, 31, 1):
    orig_img_path = origimgsdir + str(i) + ".tif"
    mask_img_path = maskimgsdir + str(i) + ".png"

    origImg = cv2.imread(orig_img_path)
    origGray = cv2.cvtColor(origImg, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(origGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    origRatio = np.count_nonzero(thresh) / thresh.size

    maskImg = cv2.imread(mask_img_path)
    maskGray = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)
    maskRatio = np.count_nonzero(maskGray) / maskGray.size

    print (origRatio, ' - ', maskRatio)


print (lstMaskRatio)
print (lstOTSURatio)