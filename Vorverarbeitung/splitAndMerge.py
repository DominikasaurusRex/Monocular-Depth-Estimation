import cv2;
import numpy as np;

windowX = 150;
windowY = 150;

def mergeImage(windowList, sizeY, sizeX):
    img = [];
    
    height = (int(sizeY / windowY) + 1);
    width = (int(sizeX / windowX) + 1);
    
    for y in range(0, height-1, 1):
        row = windowList[y*width];
        for x in range(1, width-1, 1):
            row = np.concatenate((row, windowList[x + (y*width)]), axis= 1);
        
        if y== 0:
            img = row;
        else:
            img = np.concatenate((img, row), axis=0);

    mergeImg = np.array(img);
    return mergeImg;
            
def splitImage(image):
    blockList = []
    for allY in range(0, image.shape[0], windowY):
        for allX in range(0, image.shape[1], windowX):
            block = image[allY : allY+windowY, allX : allX+windowX];
            blockList.append(block);    

    return blockList;

#if __name__ == '__main__':

#    image = cv2.imread("D:/Bilder/test.jpg");
    #print("orginal");
    #print(image);
#    imageblocks = splitImage(image);

#    newImage = mergeImage(imageblocks, image.shape[0], image.shape[1]);
    
#    cv2.imshow('Test', newImage);
#    cv2.waitKey(0);