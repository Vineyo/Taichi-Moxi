from Taichi_Moxi import *
import random

ti.init(arch=ti.vulkan)#,kernel_profiler=True)

res=512
moxi=Taichi_Moxi(res,gravity=1)

window=ti.ui.Window("taichi_moxi",(res,res))#,fast_gui=True)
canvas=window.get_canvas()
moxi.setCurrentColor(0.1, 0.3, 0.25,0.5)
moxi.setPigmentFlowRate(2)

# moxi.set_edge_parameters(q1=0.1, q2=.6, q3=0.1)
# ti.clear_kernel_profile_info()

# for i in range(1000):
radius=0.03
r=0.1
g=0.3
b=0.25
a=0.5
while window.running:
    event=window.get_events()
    radius = window.GUI.slider_float("brush radius", radius,0.0,0.1)
    r=window.GUI.slider_float("R", r,0.0,1.0)
    g=window.GUI.slider_float("G", g,0.0,1.0)
    b=window.GUI.slider_float("B", b,0.0,1.0)
    a=window.GUI.slider_float("A", a,0.0,1.0)
    window.GUI.text("Press space and drag to paint")
    moxi.setCurrentColor(r,g,b,a)
      
    if(window.is_pressed(ti.ui.LMB) and window.is_pressed(' ')):
        moxi.setCursor(window.get_cursor_pos()[0],window.get_cursor_pos()[1])
        moxi.drawStrok(radius)
    moxi.update()

    moxi.render()
    canvas.set_image(moxi.FrameBuffer)

    window.show()

# ti.print_kernel_profile_info()