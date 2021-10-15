from Taichi_Moxi import *
import random
res=512
moxi=Taichi_Moxi()

gui=ti.GUI("taichi_moxi",(res,res))
moxi.setCurrentColor(0.9, 0.9, 0.25,0.5)
moxi.setPigmentFlowRate(2)
moxi.set_edge_parameters(-0.1,0.4,0.5)
# moxi.Water_Surface[50,50]=1.0
while gui.running:
    gui.get_event()

    if(gui.is_pressed(ti.GUI.RMB)):
        moxi.setCurrentColor(random.random(),random.random(),random.random(),random.random())
        
    if(gui.is_pressed(ti.GUI.LMB)):
        moxi.setCursor(gui.get_cursor_pos()[0],gui.get_cursor_pos()[1])
        moxi.drawStrok()
    moxi.update()
    moxi.render()
    gui.set_image(moxi.FrameBuffer)
    gui.show()
