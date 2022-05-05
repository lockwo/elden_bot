import keyboard
import time

for i in range(10):
	keyboard.start_recording()
	time.sleep(0.1)
	x = keyboard.stop_recording()
	print([i.name for i in x])
