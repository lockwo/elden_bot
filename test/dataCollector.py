import time
import cv2
import mss
import numpy as np
import sys
import os
import keyboard

if __name__ == '__main__':
    if len(sys.argv)>1:
        folderString=sys.argv[1]
    else:
        folderString="test_images"
        
    path = os.path.join(os.getcwd(), folderString)
    if not os.path.isdir(path):
        os.mkdir(path)
    
    with mss.mss() as sct:
        try:
            count=0
            while True:
                # Part of the screen to capture
                monitor = {"top": 50, "left": 0, "width": 2048, "height": 1152}

                # Get raw pixels from the screen, save it to a Numpy array
                img = np.array(sct.grab(monitor))

                ratio = 240 / 2048
                new_height = int(1152 * ratio)

                img = cv2.resize(img, (240, new_height), interpolation = cv2.INTER_AREA)
                cv2.imwrite(f'{folderString}/img{count}.png', img)
                
                count+=1
        except KeyboardInterrupt:
            print("cum")

