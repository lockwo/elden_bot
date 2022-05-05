import time
import cv2
import mss
import numpy

with mss.mss() as sct:
  # Part of the screen to capture
  monitor = {"top": 50, "left": 0, "width": 2048, "height": 1152}

  # Get raw pixels from the screen, save it to a Numpy array
  img = numpy.array(sct.grab(monitor))

  ratio = 240 / 2048
  new_height = int(1152 * ratio)

  img = cv2.resize(img, (240, new_height), interpolation = cv2.INTER_AREA)
  # Display the picture
  cv2.imwrite("test_images/Numpy_test.png", img)

  # Display the picture in grayscale
  # cv2.imshow('OpenCV/Numpy grayscale',
  #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))


