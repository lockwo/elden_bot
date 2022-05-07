import cv2
import os

import numpy as np
import time
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

if __name__=='__main__':
    imgs=load_images_from_folder(os.path.join(os.getcwd(), 'test_images'))
    
    left=908
    down=495
    width=8
    height=1067
    '''
    horizontal: 904
    vertical: 495
    width: 15
    height: 1067
    '''
    
    for i in range(len(imgs)):
        
        
        init=time.time()
        img2=imgs[i][left:left+width,down:down+height]
        ret,threshb = cv2.threshold(img2[:,:,0],190,255,cv2.THRESH_BINARY)
        ret,threshg = cv2.threshold(img2[:,:,1],190,255,cv2.THRESH_BINARY)
        ret,threshr = cv2.threshold(img2[:,:,2],190,255,cv2.THRESH_BINARY)
        summed=threshb+threshg+threshr
        value=np.max(np.argmax(summed,axis=1))
        ending=time.time()
        print(value)
        print(ending-init)
        cv2.imshow('original',imgs[i])
        cv2.imshow('roi',img2)
        cv2.imshow('redChannel',threshr)
        cv2.imshow('blueChannel',threshb)
        cv2.imshow('greenChannel',threshg)
        
        
        key=cv2.waitKey(0)
        
    
    