import cv2
import os
import numpy as np

binaryImagePath = 'results/mask_TCGA-21-5786-01Z-00-DX1.png'

img = cv2.imread(binaryImagePath, cv2.IMREAD_GRAYSCALE)

imgColor = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


im2, cnts, hierarchy = cv2.findContours(img.copy(), cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)

print (len(cnts))

for c in cnts:
    M = cv2.moments(c)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    area = cv2.contourArea(c)

    if area > img.size * 0.8:
        print( area )
        continue

    cv2.drawContours(imgColor, [c], -1, (0, 255, 0), 2)
    cv2.circle(imgColor, (cX, cY), 4, (0, 0, 255), -1)

    #cv2.imshow('img', imgColor)
    #cv2.waitKey(0)


cv2.imwrite('dst.png', imgColor);