import time
import cv2
import mss
import numpy as np

def index_to_key(inds):
    vals = ["W", "A", "S", "D", "F", "R", "U", "I", "O", "P", " "]
    keys = []
    for i in range(len(inds)):
    	if inds[i] == 1:
    		keys.append(vals[i])
    return keys

batch = 100
init = 0
end = batch
for i in range(1000):
	if i == init:
		ins = np.load(f'test_images/{init}_{end}.npy')
		images = np.load(f'test_images/img{init}_{end}.npy')
	if i == end:
		init += 100
		end += 100
		ins = np.load(f'test_images/{init}_{end}.npy')
		images = np.load(f'test_images/img{init}_{end}.npy')

	keys = index_to_key(ins[i - init])
	print(keys)
	img = images[i - init]
	cv2.imshow('test', img)
	#input()
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
