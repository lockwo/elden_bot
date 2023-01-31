import cv2
import numpy as np
import sys


def index_to_key(inds):
    vals = ["W", "A", "S", "D", "F", "R", "U", "I", "O", "P", " "]
    keys = []
    for i in range(len(inds)):
        if inds[i] == 1:
            keys.append(vals[i])
    return keys


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folderString = sys.argv[1]
    else:
        folderString = "test_images1"
    batch = 100
    init = 0
    end = batch
    i = 0
    while True:
        try:
            if i == init:
                ins = np.load(f"{folderString}/{init}_{end}.npy")
                images = np.load(f"{folderString}/img{init}_{end}.npy")
                rewards = np.load(f"{folderString}/r{init}_{end}.npy")
            if i == end:
                init += 100
                end += 100
                ins = np.load(f"{folderString}/{init}_{end}.npy")
                images = np.load(f"{folderString}/img{init}_{end}.npy")
                rewards = np.load(f"{folderString}/r{init}_{end}.npy")
        except FileNotFoundError:
            print("End of Playback")
            break

        keys = index_to_key(ins[i - init])
        print(keys, rewards[i - init])
        img = images[i - init]
        cv2.imshow("test", img)
        # input()
        i += 1
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
