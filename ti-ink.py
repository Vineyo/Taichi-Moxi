import taichi as ti
import numpy as np
import matplotlib.image as mpig
import taichi_glsl as tg
ti.init(arch=ti.cuda)

res = 512
brushRadius = 0.05
omiga = 0.5
alpha=0.3
_Surface = ti.Vector.field(4, dtype=float, shape=(res, res))
_Flow = ti.Vector.field(9, dtype=float, shape=(res, res))
_FlowNext = ti.Vector.field(9, dtype=float, shape=(res, res))
_Feq = ti.Vector.field(9, dtype=float, shape=(res, res))
_Fixture = ti.Vector.field(3, dtype=float, shape=(res, res))
_BackGroundLayer = ti.Vector.field(3, dtype=float, shape=(res, res))
_FrameBuffer = ti.Vector.field(3, dtype=float, shape=(res, res))
cursor = ti.field(float, shape=2)
currentColor = ti.Vector([0.0, 0.0, 0.0,1.0])
_paper = ti.field(dtype=float, shape=(res, res))
_fibers = ti.field(dtype=float, shape=(res, res))
_rou = ti.field(float, shape=(res, res))

_paper.from_numpy(mpig.imread("paper_512.png")[:,:,0])
_fibers.from_numpy(mpig.imread("fibers_512.png")[:,:,0])

e = ti.Vector.field(2, dtype=int, shape=9)
e[0] = ti.Vector([0, 0])
e[1] = ti.Vector([0, 1])
e[2] = ti.Vector([-1, 0])
e[3] = ti.Vector([0, -1])
e[4] = ti.Vector([1, 0])
e[5] = ti.Vector([1, 1])
e[6] = ti.Vector([-1, 1])
e[7] = ti.Vector([-1, -1])
e[8] = ti.Vector([1, -1])

w = ti.field(dtype=float, shape=9)
w[0] = 4.0/9.0
w[1] = 1.0/9.0
w[2] = 1.0/9.0
w[3] = 1.0/9.0
w[4] = 1.0/9.0
w[5] = 1.0/36.0
w[6] = 1.0/36.0
w[7] = 1.0/36.0
w[8] = 1.0/36.0


@ti.kernel
def update_rou():
    for P in ti.grouped(_rou):
        _rou[P] = _Flow[P].sum()


@ti.kernel
def update_FlowNext():
    for P in ti.grouped(_FlowNext):
        for i in ti.static(range(9)):
            prePos = (P[0]-e[i][0], P[1]-e[i][1])
            _FlowNext[P][i] = omiga*(_Feq[prePos][i]-_Flow[prePos][i])+_Feq[prePos][i]


@ti.kernel
def update_flow():
    for P in ti.grouped(_Flow):
        _Flow[P] = _FlowNext[P]


@ti.kernel
def update_Feq():
    for P in ti.grouped(_Feq):
        u = ti.Vector([0.0, 0.0])
        for j in ti.static(range(9)):
            u += _Flow[P][j]*e[j]
        for i in ti.static(range(9)):
            _Feq[P][i] = w[i]*(_rou[P] + tg.scalar.smoothstep(0,alpha,_rou[P])*(3 * e[i].dot(u) +
                               4.5 * (e[i].dot(u))**2 - 1.5 * u.dot(u)))


@ti.func
def clamp(v, vmin, vmax):
    return min(vmax, max(vmin, v))


@ti.kernel
def fill_BG():
    for p in ti.grouped(_BackGroundLayer):
        _BackGroundLayer[p][0] = 1.0
        _BackGroundLayer[p][1] = 1.0
        _BackGroundLayer[p][2] = 1.0


@ti.kernel
def fill_PL(r: float, g: float, b: float, a: float):
    for P in ti.grouped(_Surface):
        _Surface[P][0] = r
        _Surface[P][1] = g
        _Surface[P][2] = b
        _Surface[P][3] = a


@ti.kernel
def render():
    for P in ti.grouped(_FrameBuffer):
        for i in ti.static(range(3)):
            _FrameBuffer[P][i] = (
                -_BackGroundLayer[P][i])*_rou[P] + _BackGroundLayer[P][i]


@ti.kernel
def drawStrok(radius: ti.f32):
    center = ti.Vector([cursor[0], cursor[1]])
    for P in ti.grouped(_Surface):
        dis = (P/res-center).norm()
        if dis < radius:
            dens = clamp(1-dis/radius, 0, 1)
            _Surface[P] = (currentColor - _Surface[P])*dens + _Surface[P]
            _Surface[P][3] = tg.scalar.clamp(_Surface[P][3]+dens, 0, 0.8)


@ti.kernel
def surface_to_flow():
    for P in ti.grouped(_Flow):
        _Flow[P][0] += _Surface[P][3]
        _Surface[P][3]=0


gui = ti.GUI("ti-moxi", (res, res))
fill_BG()

cursor[0] = 0.5
cursor[1] = 0.5

while gui.running:
    
    gui.get_event()
    cursor[0] = gui.get_cursor_pos()[0]
    cursor[1] = gui.get_cursor_pos()[1]
    if(gui.is_pressed(ti.GUI.LMB)):
        drawStrok(0.03)
        surface_to_flow()
    if(gui.is_pressed(ti.GUI.RMB)):
        fill_PL(0.0, 0.0, 0.0, 0.0)
    update_rou()
    update_Feq()
    update_FlowNext()
    update_flow()
    render()
    gui.set_image(_FrameBuffer)
    gui.show()
