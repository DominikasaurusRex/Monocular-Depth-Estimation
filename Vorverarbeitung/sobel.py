#encoding: utf-8

import cv2
import os
import glob
import numpy as np;

def sobel(image, kernelSize, direction):
    im = image.astype(np.float);
    width, hight, c = im.shape;
    
    if c > 1:
        img = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2];  
    else:
        img = im;
    
    assert(kernelSize == 3 or kernelSize == 5);
     
    if kernelSize == 3:
        kernelHorizontal = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype = np.float);
        kernelVertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float);
    else:
        kernelHorizontal = np.array([[1, 2, 0, -2, -1], 
                   [4, 8, 0, -8, -4], 
                   [6, 12, 0, -12, -6],
                   [4, 8, 0, -8, -4],
                   [1, 2, 0, -2, -1]], dtype = np.float);
        kernelVertical = np.array([[-1, -4, -6, -4, -1], 
                   [-2, -8, -12, -8, -2],
                   [0, 0, 0, 0, 0], 
                   [2, 8, 12, 8, 2],
                   [1, 4, 6, 4, 1]], dtype = np.float);

    if direction == 0:               
        img = cv2.filter2D(img, -1, kernelHorizontal);
    elif direction == 1:
        img = cv2.filter2D(img, -1, kernelVertical);
    
    #g = np.sqrt(gx * gx + gy * gy);
    #g *= 255.0 / np.max(img);
    
    return img;

if __name__ == '__main__':
    outputfolder = 'D:\\Bilder\\nyu_datasets_2\\'
    
    filenames = []
    images = []
    depths = []
    img = []
    outputpath = []
    outputpath2 = []
    
    
    for filename in os.listdir('C:\\Users\\Domi\\workspace\\Monocular-Depth-Estimation\\data\\nyu_datasets'):
        filenames.append(filename);
    
    for i in range(0, len(filenames), 2):
        outputpath.append(outputfolder + filenames[i])
        outputpath2.append(outputfolder + filenames[i+1])
    
    for files in glob.glob('C:\\Users\\Domi\\workspace\\Monocular-Depth-Estimation\\data\\nyu_datasets\\*.jpg'):
        image = cv2.imread(files)
        images.append(image)
    
    for depth_map in glob.glob('C:\\Users\\Domi\\workspace\\Monocular-Depth-Estimation\\data\\nyu_datasets\\*.png'):
        depth = cv2.imread(depth_map)
        depths.append(depth)
        
    for i in range(0, len(images), 1):
        img.append(sobel(images[i], 3, 0))
        cv2.imwrite(outputpath[i], img[i])
        cv2.imwrite(outputpath2[i], depths[i])
    

    #cv2.imshow('Test', img);
    #cv2.waitKey(0);
    
    
