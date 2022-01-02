from taichi.lang import collect_kernel_profile_metrics
from Taichi_Moxi import *

pi=3.141592653589793
ti.init(arch=ti.vulkan)  # ,kernel_profiler=True)

res = 640
moxi = Taichi_Moxi(res)

window = ti.ui.Window("taichi_moxi", (res, res))
canvas = window.get_canvas()

radius = 0.1
color_old = (0.1, 0.3, 0.25)
q1_old, q2_old, q3_old = (0.1, 0.6, 0.1)
a_old = 0.5
brush_wetness = 0.5
flow_rate_old = 4
paper_dryness_old = 0.5
activated_gravity_old = False
softness = 0.5

brush_tip=ti.Vector.field(2, dtype=ti.f32, shape=32)

@ti.kernel
def get_brush_tip(x: ti.f32, y: ti.f32, radius: ti.f32):
    r=radius**2
    for i in ti.static(range(16)):
        brush_tip[2*i][0] = x + r * ti.cos(2*pi*i/16)
        brush_tip[2*i][1] = y + r * ti.sin(2*pi*i/16)
        brush_tip[2*i-1][0] = brush_tip[2*i][0]
        brush_tip[2*i-1][1] = brush_tip[2*i][1]

# ti.clear_kernel_profile_info()

# for i in range(1000):
while window.running:
    events = window.get_events()
    cursor_pos = window.get_cursor_pos()
    window.GUI.begin("Tools", .0, .0, 0.4, 0.3)
    window.GUI.text("Press space and drag to paint")
    window.GUI.text("Press 'Alt' and click to pick color")
    color_new = window.GUI.color_edit_3("brush color", color_old)
    radius = window.GUI.slider_float("brush radius", radius, 0.0, 0.3)
    a_new = window.GUI.slider_float("A", a_old, 0.0, 1.0)
    softness = window.GUI.slider_float("softness", softness, 0.0, 1.0)
    brush_wetness = window.GUI.slider_float(
        "brush wetness", brush_wetness, 0.0, 1.0)
    flow_rate_new = window.GUI.slider_float(
        "flow rate", flow_rate_old, 0.0, 4.0)
    paper_dryness_new = window.GUI.slider_float(
        "paper dryness", paper_dryness_old, 0.0, 0.99)
    q1_new = window.GUI.slider_float("q1", q1_old, -0.02, 0.02)
    q2_new = window.GUI.slider_float("q2", q2_old, 0.0, 2.0)
    q3_new = window.GUI.slider_float("q3", q3_old, 0.0, 2.0)
    activated_gravity_new = window.GUI.checkbox(
        "activate gravity", activated_gravity_old)
    clear_is_clicked = window.GUI.button("Clear")
    dry_is_clicked = window.GUI.button("Dry")
    save_is_clicked = window.GUI.button("Save(Enter name in console)")
    window.GUI.end()

    if q1_new != q1_old or q2_new != q2_old or q3_new != q3_old:
        moxi.set_edge_parameters(q1=q1_new, q2=q2_new, q3=q3_new)
        moxi.update_sigma()
        q1_old, q2_old, q3_old = q1_new, q2_new, q3_new
    if color_new != color_old or a_new != a_old:
        color_old = color_new
        a_old = a_new
        moxi.setCurrentColor(color_new[0], color_new[1], color_new[2], a_new)
    if flow_rate_new != flow_rate_old:
        flow_rate_old = flow_rate_new
        moxi.pigment_flow_rate[None] = flow_rate_old
    if paper_dryness_new != paper_dryness_old:
        moxi.paper_dryness[None] = paper_dryness_old
        paper_dryness_old = paper_dryness_new
        
    if activated_gravity_new != activated_gravity_old:
        activated_gravity_old = activated_gravity_new
        moxi.activated_gravity[None] = activated_gravity_new and 1 or 0

    if clear_is_clicked:
        moxi.clear()
    if dry_is_clicked:
        moxi.dry()
    if save_is_clicked:
        name = input("Enter file name: ")
        name += ".png"
        moxi.save(name)
    if(window.is_pressed(ti.ui.LMB) and window.is_pressed(' ')):
        moxi.setCursor(cursor_pos[0], cursor_pos[1])
        moxi.drawStrok(radius, brush_wetness,softness)
    if(window.is_pressed(ti.ui.LMB) and window.is_pressed(ti.ui.ALT)):
        color=moxi.FrameBuffer[int(cursor_pos[0]*res),int(cursor_pos[1]*res)]
        color_old=(color[0],color[1],color[2])
        moxi.setCurrentColor(color_new[0], color_new[1], color_new[2], a_new)
    moxi.update()

    moxi.render()
    get_brush_tip(cursor_pos[0], cursor_pos[1], radius)
    canvas.set_image(moxi.FrameBuffer)
    canvas.lines(brush_tip, 0.002)

    window.show()

# ti.print_kernel_profile_info()
