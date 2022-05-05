import keyboard
import time
import pyautogui
import pydirectinput

fps = 2000

time.sleep(4)

for i in range(3 * fps):
	pydirectinput.keyDown("w")
	pydirectinput.keyUp("w")
	#time.sleep(1/fps)

#pyautogui.press("esc")

#for i in range(1):
#	pyautogui.press("w")

#for i in range(100):
	#keyboard.press("w")

'''
for i in range(10000):
	keyboard.press("w")
	keyboard.release("w")
	time.sleep(1/fps)
'''
