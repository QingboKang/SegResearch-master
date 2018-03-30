import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import normalizeStaining
import evaluation
from PIL import Image

def CountContoursStddev(binImg):
    colorImg = cv2.cvtColor(binImg, cv2.COLOR_GRAY2BGR)
    im2, cnts, hierchy = cv2.findContours( binImg.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE )

    lstAreas = []
    area_count = 0;
    for c in cnts:
        area = cv2.contourArea(c)
        if area < binImg.size * 0.2:
            area_count += 1
            lstAreas.append(area)
            color = list(np.random.random(size=3) * 256)
            cv2.drawContours(colorImg, [c], -1, color, 2)

    arr = np.array(lstAreas)
    area_std = np.std(arr)
    return area_count, area_std

def CountBlackHoles( binImg ):
    colorImg = cv2.cvtColor(binImg, cv2.COLOR_GRAY2BGR)
    im2, cnts, hierchy = cv2.findContours( binImg.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE )

    lstAreas = []
    area_count = 0;
    for c in cnts:
        area = cv2.contourArea(c)
        if area < binImg.size * 0.2:
            area_count += 1
            lstAreas.append(area)
            color = list(np.random.random(size=3) * 256)
            cv2.drawContours(colorImg, [c], -1, color, 2)

    arr = np.array(lstAreas)
    area_std = np.std(arr)
   # print (area_std)
   # print (area_count, "/", len(cnts))
    cv2.imshow("contour", colorImg)
    return area_count, area_std


def ImageWatershed(srcImageName, dstImageName, dstMarkName):
    img = cv2.imread(srcImageName)

    '''Color Normalization'''
    img = img.astype('float32')
    img = normalizeStaining.normalizeStaining(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ddepth = cv2.CV_32F
    dx = cv2.Sobel(gray, ddepth, 1, 0)
    dy = cv2.Sobel(gray, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    thresh1 = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 501, 31)

   # thresh = cv2.addWeighted(thresh1, 0.25, thresh2, 0.75, 0)

   # cv2.imshow('gray', gray)
   # cv2.imshow('binary', thresh)

#    cv2.imwrite('binary.png', thresh);

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

   # cv2.imshow('opening', opening);
    #cv2.imwrite('opening.png', opening);

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

  #  cv2.imshow("background", sure_bg);
    #cv2.imwrite('bg.png', sure_bg);

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    cv2.imshow('dist_transform', dist_transform);

    ret, sure_fg = cv2.threshold(dist_transform, 0.28*dist_transform.max(), 255, 0)


    cv2.imshow('foreground', sure_fg);
    #cv2.imwrite('fg.png', sure_fg);

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
   # bin_mask[markers == -1] = 255
    bin_mask[markers == 1] = 0
#    bin_mask[bin_mask == -1] = 0;
    bin_mask = bin_mask.astype(np.uint8)
    #print (bin_mask.shape )

   # cv2.imshow('test', img);
   # cv2.imshow('bin_mask', bin_mask);
    cv2.imwrite( dstMarkName, bin_mask);

    ratio = evaluation.get_white_ratio(bin_mask)
    area_count, area_mean, area_std = evaluation.count_contours_stddev(bin_mask)

  #  cv2.waitKey(0);
 #   cv2.imwrite(dstImageName, img);
    cv2.destroyAllWindows();

    return ratio, area_count, area_mean, area_std;


def ImageWatershed1(srcImageName, dstImageName, dstMarkName):
    img = cv2.imread(srcImageName)

    '''Color Normalization'''
 #   img = img.astype('float32')
  #  img = normalizeStaining.normalizeStaining(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ddepth = cv2.CV_32F
    dx = cv2.Sobel(gray, ddepth, 1, 0)
    dy = cv2.Sobel(gray, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    thresh1 = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 501, 11)

   # thresh = cv2.addWeighted(thresh1, 0.25, thresh2, 0.75, 0)

   # cv2.imshow('gray', gray)
    cv2.imshow('binary', thresh)

#    cv2.imwrite('binary.png', thresh);

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    cv2.imshow('opening', opening);
    #cv2.imwrite('opening.png', opening);

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    cv2.imshow("background", sure_bg);
    #cv2.imwrite('bg.png', sure_bg);

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    cv2.imshow('dist_transform', dist_transform);

    ret, sure_fg = cv2.threshold(dist_transform, 0.25*dist_transform.max(), 255, 0)


    cv2.imshow('foreground', sure_fg);
    #cv2.imwrite('fg.png', sure_fg);

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
   # bin_mask[markers == -1] = 255
    bin_mask[markers == 1] = 0
#    bin_mask[bin_mask == -1] = 0;
    bin_mask = bin_mask.astype(np.uint8)
    #print (bin_mask.shape )

    cv2.imshow('test', img);
    cv2.imshow('bin_mask', bin_mask);
    cv2.imwrite( dstMarkName, bin_mask);

    count_holes, std_holes = CountBlackHoles(bin_mask)

    #cv2.waitKey(0);
    cv2.imwrite(dstImageName, img);
    cv2.destroyAllWindows()

    return count_holes, std_holes;

tif_dir = '../Tissue images/'
# tif_dir = '../bad/'
bad_dir = '../bad/'
result_dir = 'results/'

tif_files = os.listdir(tif_dir)
count = 1
for file in tif_files:
    if not os.path.isdir(file):
        tiffilename = tif_dir + file
        resultFileName = result_dir + file[0:file.find('.tif')] + '.png'
        maskFileName = result_dir + file[0:file.find('.tif')] + '.png'

        ratio, area_count, area_mean, area_std = ImageWatershed(tiffilename, resultFileName, maskFileName)

        count += 1
        print(maskFileName, " ", str(ratio), " ", str(area_count), " ", str(area_mean), " ", str(area_std))
        '''
        count_holes, std_holes = ImageWatershed1(tiffilename, resultFileName, maskFileName)
        count += 1
        '''




