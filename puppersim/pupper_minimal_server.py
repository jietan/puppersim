import pybullet as p
import time

GRAVITY = -10

cid = p.connect(p.GUI_SERVER)
# disable keyboard shortcuts that allow you to toggle things like shadows (s) or wire-frame (w)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS,0) # s and w
# disable shadows to potentially increase speed
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
while p.isConnected():
    p.setGravity(0,0,GRAVITY)
    time.sleep(0.01)
