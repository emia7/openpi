import keyboard
import time

def start_collect():
    keyboard.send('up')
    time.sleep(0.05)
    keyboard.send('enter')

def stop_collect():
    keyboard.send('ctrl+c')

keyboard.add_hotkey('a', start_collect)
keyboard.add_hotkey('c', stop_collect)

keyboard.wait()