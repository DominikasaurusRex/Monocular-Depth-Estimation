'''
Created on 25.04.2017

@author: Domi
'''

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
img = cv2.imread('C:/Users/Domi/Downloads/KompetenzradDomi.png', cv2.IMREAD_GRAYSCALE)
edge = cv2.Canny(img, 100, 200)
cv2.imshow('image', img)
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

hello = tf.constant('HELLO SHIT')
sess = tf.Session()
print(sess.run(hello))


node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

