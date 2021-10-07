import taichi
from Taichi_Moxi import *
import random
res=512
moxi=Taichi_Moxi(res)

gui=ti.GUI("taichi_moxi",(res,res))
cursor=ti.Vector([0.5,0.5])
moxi.setCurrentColor(0.9, 0.9, 0.25,0.5)

while gui.running:
    gui.get_event()

    if(gui.is_pressed(ti.GUI.RMB)):
        
        moxi.setCurrentColor(random.random(),random.random(),random.random(),random.random())
    if(gui.is_pressed(ti.GUI.LMB)):
        # cursor[0] = gui.get_cursor_pos()[0]
        # cursor[1] = gui.get_cursor_pos()[1]
        moxi.setCursor(gui.get_cursor_pos()[0],gui.get_cursor_pos()[1])
        moxi.drawStrok()
    moxi.update()
    moxi.render()
    gui.set_image(moxi.FrameBuffer)
    gui.show()
