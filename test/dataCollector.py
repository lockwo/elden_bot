import time
import cv2
import mss
import numpy as np
import sys
import os
import keyboard
import win32api as wapi
import time

def make_index():
    vals = ["W", "A", "S", "D", "F", "R", "U", "I", "O", "P", " "]
    inds = dict()
    for i in range(len(vals)):
        inds[vals[i]] = i
    return inds

indexes = make_index()

# Citation: Box Of Hats (https://github.com/Box-Of-Hats)
keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
        # Space key
        if wapi.GetAsyncKeyState(0):
            keys.append(" ")
    return keys

def end():
    print("Done")
    ending = time.time()
    duration = ending - init
    print(f'Average FPS: {count/duration}')



def keys_to_array(keys):
    # [W, A, S, D, F, R, U, I, O, P, SPACE]
    x = np.zeros(shape=(11), dtype=np.uint8)
    for key in keys:
        if key in indexes:
            x[indexes[key]] = 1
    return x

if __name__ == '__main__':
    if len(sys.argv) > 1:
        folderString = sys.argv[1]
    else:
        folderString = "test_images"
        
    path = os.path.join(os.getcwd(), folderString)
    
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        temp = 1
        while os.path.isdir(path):
            path = os.path.join(os.getcwd(), folderString + str(temp))   
            temp += 1
        folderString += str(temp-1)
        path = os.path.join(os.getcwd(), folderString)   
        os.mkdir(path)
    
    init_count = 0
    save_batch_size = 100
    key_ins = np.zeros(shape=(save_batch_size, 11), dtype=np.uint8)
    width = 244
    #ratio =  width / 2048
    new_height = 138 #int(1152 * ratio)
    images = np.zeros(shape=(save_batch_size, new_height, width), dtype=np.uint8)
    time.sleep(5)
    init = time.time()
    with mss.mss() as sct:
        try:
            count = 0
            while True:
                # Part of the screen to capture
                monitor = {"top": 50, "left": 0, "width": 2048, "height": 1152}

                # Get raw pixels from the screen, save it to a Numpy array
                img = np.array(sct.grab(monitor))

                img = cv2.resize(img, (width, new_height), interpolation = cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                #img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                images[count - init_count] = img
                #cv2.imwrite(f'{folderString}/img{count}.png', img)

                keys = key_check()
                key_ins[count - init_count] = keys_to_array(keys)

                count += 1
                if count % save_batch_size == 0 and count > 10:
                    np.save(f'{folderString}/{init_count}_{count}', key_ins)
                    np.save(f'{folderString}/img{init_count}_{count}', images)
                    key_ins = np.zeros(shape=(save_batch_size, 11), dtype=np.uint8)
                    images = np.zeros(shape=(save_batch_size, new_height, width), dtype=np.uint8)
                    init_count = count

        except KeyboardInterrupt:
            end()
