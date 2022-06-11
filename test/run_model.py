import tensorflow as tf
import time
import cv2
import mss
import numpy as np
import sys
import os
import keyboard
import win32api as wapi
import time
from collections import deque
import pyautogui
import pydirectinput
import tensorflow as tf

# Better performance
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def press_keys(p):
	vals = ["w", "a", "s", "d", "f", "r", "u", "i", "o", "p", " "]
	for i in range(len(p[0])):
		if p[0][i] > 0.5:
			pydirectinput.keyDown(vals[i], _pause=False)
			print(vals[i])
		else:
			pydirectinput.keyUp(vals[i], _pause=False)

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
	init = tf.keras.initializers.VarianceScaling(scale=2)
	inputs = tf.keras.layers.Input(shape=(244, 138, 3))
	num_actions = 11
	x = tf.keras.layers.Conv2D(32, 8, strides=(4,4), activation='relu', kernel_initializer=init)(inputs)
	x = tf.keras.layers.Conv2D(64, 4, strides=(3,3), activation='relu', kernel_initializer=init)(x)
	x = tf.keras.layers.Conv2D(64, 3, strides=(1,1), activation='relu', kernel_initializer=init)(x)
	#x = tf.keras.layers.Conv2D(128, 3, strides=(1,1), activation='relu')(x)
	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dropout(0.3)(x)
	#x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=init)(x)
	#x = tf.keras.layers.Dropout(0.3)(x)
	x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=init)(x)
	x = tf.keras.layers.Dropout(0.3)(x)
	x = tf.keras.layers.Dense(num_actions, activation='sigmoid')(x)
	model = tf.keras.models.Model(inputs=inputs, outputs=x)
	#checkpoint_filepath = './small_classweight/checkpoint'
	checkpoint_filepath = './video_test/checkpoint'
	model.load_weights(checkpoint_filepath)

	model(np.random.uniform(-1, 1, (1, 244, 138, 3)))

	init_count = 0
	width = 244
	#ratio =  width / 2048
	new_height = 138 #int(1152 * ratio)
	left = 908
	down = 495
	width2 = 8
	height = 1067

	max_health = None

	for i in range(5):
		print(i)
		time.sleep(1)
	init = time.time()
	inputs = deque(maxlen=3)
	with mss.mss() as sct:
		try:
			count = 0
			while True:
				# Part of the screen to capture
				monitor = {"top": 50, "left": 0, "width": 2048, "height": 1152}

				# Get raw pixels from the screen, save it to a Numpy array
				img = np.array(sct.grab(monitor))
				
				img2 = img[left:left+width,down:down+height]
				ret,threshb = cv2.threshold(img2[:,:,0],190,255,cv2.THRESH_BINARY)
				ret,threshg = cv2.threshold(img2[:,:,1],190,255,cv2.THRESH_BINARY)
				ret,threshr = cv2.threshold(img2[:,:,2],190,255,cv2.THRESH_BINARY)
				summed = threshb+threshg+threshr
				value = np.max(np.argmax(summed,axis=1))
				if count == 0:
					max_health = value
				if max_health < 1:
					max_health = 1000

				value = -value/max_health
				img = cv2.resize(img, (width, new_height), interpolation = cv2.INTER_AREA)
				img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
				if count == 0:
					for _ in range(3):
						inputs.append(img)
				else:
					inputs.append(img)
				# (old, mid, newest)
				img = np.array(inputs)
				img = np.expand_dims(img, axis=0).transpose(0, 3, 2, 1)

				pred = model(img)
				press_keys(pred)
				count += 1


		except KeyboardInterrupt:
			end()
