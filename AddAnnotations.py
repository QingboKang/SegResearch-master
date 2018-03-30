import cv2
import os
import numpy as np

def AddOrigMask(orig_img_filename, mask_img_filename, dst_img_filename):
    orig_img = cv2.imread(orig_img_filename)
    mask_img = cv2.imread(mask_img_filename)

    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY);

    #mask_img[mask_img == 255] = 40;

    #print (mask_img.shape)
    #print (orig_img.shape)

    orig_img[mask_img == 255] = [255,0,0]

    cv2.imwrite( dst_img_filename, orig_img);
    return ;

tif_dir = '../Tissue images/'
mask_dir = '../Mask/'
dst_dir = '../OrigMask/'
png_files = os.listdir(tif_dir);
for file in png_files:
    if not os.path.isdir(file):
        tiffilename = tif_dir + file;
        pngfilename = mask_dir + file[0:file.find('.tif')] + '.png';
        dstfilename = dst_dir + file[0:file.find('.tif')] + '.png';

        AddOrigMask(tiffilename, pngfilename, dstfilename)

