import cv2
import numpy as np

def insertionSort(pixel):

    temp = [];
    
    for i in range(0, 9, 1):
        temp = pixel[i];
        
        j = i-1;
        while j >= 0 and np.any(temp < pixel[j]):
            pixel[j+1] = pixel[j];
            j = j - 1;
        
        pixel[j+1] = temp;

    return pixel;

def medianFilter(image):
    pixel = [None] * 9;

    dst = image.copy();
    
    for y in range(1, image.shape[0]-1, 1):
        for x in range(1, image.shape[1]-1, 1):
            pixel[0] = image[y-1][x-1];
            pixel[1] = image[y][x-1];
            pixel[2] = image[y+1][x-1];
            pixel[3] = image[y-1][x];
            pixel[4] = image[y][x];
            pixel[5] = image[y+1][x];
            pixel[6] = image[y-1][x+1];
            pixel[7] = image[y][x+1];
            pixel[8] = image[y+1][x+1];

            sortedPixel = np.array(insertionSort(pixel));
            dst[y][x] = pixel[4];
            
    return dst;

#if __name__ == '__main__':
#    image = cv2.imread("D:/Bilder/test1.jpg");
#    #image = cv2.imread("D:/Bilder/test2.png");
#    img = medianFilter(image);
#      
#    #print(img)
#    cv2.imshow('Test', img);
#    cv2.waitKey(0);
#    
#    pass