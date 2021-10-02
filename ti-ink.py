import taichi as ti
import matplotlib.image as mpig
import taichi_glsl as tg
from taichi_glsl.experimental_array import dtype
ti.init(arch=ti.cuda)

res = 512
brushRadius = 0.05


_Pigment_Surface = ti.Vector.field(4, dtype=float, shape=(res, res))
_Water_Surface = ti.field(float, shape=(res, res))
_Flow = ti.Vector.field(9, dtype=float, shape=(res, res))
_FlowNext = ti.Vector.field(9, dtype=float, shape=(res, res))
_Feq = ti.Vector.field(9, dtype=float, shape=(res, res))
_kapar = ti.Vector.field(9, dtype=float, shape=(res, res))
_Fixture = ti.Vector.field(4, dtype=float, shape=(res, res))
_BackGroundLayer = ti.Vector.field(3, dtype=float, shape=(res, res))
_FrameBuffer = ti.Vector.field(3, dtype=float, shape=(res, res))
cursor = ti.field(float, shape=2)
currentColor = ti.Vector.field(4, dtype=float, shape=2)
currentColor[0] = ti.Vector([0.0, 0.4, 0.25, 0.8])
currentColor[1] = ti.Vector([0.0, 0.1, 0.2, 0.0])
_paper = ti.field(dtype=float, shape=(res, res))
_fibers = ti.field(dtype=float, shape=(res, res))
_rou = ti.field(float, shape=(res, res))
_sigma = ti.field(float, shape=(res, res))
_paper.from_numpy(mpig.imread("paper_512_2.png")[:, :, 0])
_fibers.from_numpy(mpig.imread("fibers_512_2.png")[:, :, 0])
_ispinning = ti.field(dtype=int, shape=(res, res))
_kar = ti.field(dtype=float, shape=(res, res))
_Pigment_flow = ti.Vector.field(4, dtype=float, shape=(res, res))
_Pigment_flow_star = ti.Vector.field(4, dtype=float, shape=(res, res))
_FlowVelocity = ti.Vector.field(2, dtype=float, shape=(res, res))
# e = ((0, 0), (0, 1), (-1, 0), (0, -1), (1, 0),
#            (1, 1), (-1, 1), (-1, -1), (1, -1))

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

k = (0, 3, 4, 1, 2, 7, 8, 5, 6)

w = (4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0 /
     9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0)

# if(True):
#     w = (4.0/9.0, 0.8/9.0, 1.0/9.0, 1.2/9.0, 1.0 /
#          9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0)

q1 = 0.01
q2 = 0.9
q3 = 0.5


@ti.kernel
def ini_sigma():
    for P in ti.grouped(_sigma):
        _sigma[P] = q1+q2*_fibers[P]


@ti.kernel
def get_ispinning():
    for P in ti.grouped(_ispinning):
        if (_rou[P] == 0 and
            _rou[P+e[1]] < _sigma[P+e[1]] and
            _rou[P+e[2]] < _sigma[P+e[2]] and
            _rou[P+e[3]] < _sigma[P+e[3]] and
            _rou[P+e[4]] < _sigma[P+e[4]] and
            _rou[P+e[5]] < (_sigma[P+e[5]]*1.414213562373095048) and
            _rou[P+e[6]] < (_sigma[P+e[6]]*1.414213562373095048) and
            _rou[P+e[7]] < (_sigma[P+e[7]]*1.414213562373095048) and
                _rou[P+e[8]] < (_sigma[P+e[8]]*1.414213562373095048)):
            _ispinning[P] = 1
        else:
            _ispinning[P] = 0


@ti.kernel
def update_karpa():
    for P in ti.grouped(_kapar):
        for i in ti.static(range(9)):
            P_nebor = P+e[i]
            _kapar[P][i] = 0.5*(_kar[P]+_kar[P_nebor])


@ti.kernel
def ini_k():
    for P in ti.grouped(_kar):
        _kar[P] = _paper[P]


@ti.kernel
def update_k_for_pinning():
    for P in ti.grouped(_kar):
        if(_ispinning[P]):
            _kar[P] = 0.7


@ti.kernel
def update_rou():
    for P in ti.grouped(_rou):
        _rou[P] = _Flow[P].sum()*(0.995)


omiga = 0.5


@ti.kernel
def update_FlowNext():
    for P in ti.grouped(_FlowNext):
        for i in ti.static(range(9)):
            prePos = P-e[i]
            _FlowNext[P][i] = omiga * \
                (_Feq[prePos][i]-_Flow[prePos][i])+_Feq[prePos][i]


@ti.kernel
def update_flow():
    for P in ti.grouped(_Flow):
        # _Flow[P]=_FlowNext[P]
        for i in ti.static(range(9)):
            prePos = P-e[i]
            _Flow[P][i] = _kapar[P][i] * \
                (_FlowNext[P][k[i]] - _FlowNext[prePos][i]) + \
                _FlowNext[prePos][i]


@ti.kernel
def update_Feq():
    alpha = 0.5
    for P in ti.grouped(_Feq):
        _FlowVelocity[P] = ti.Vector([0.0, 0.0])
        for j in ti.static(range(1, 9)):
            _FlowVelocity[P] += _Flow[P][j]*e[j]
        for i in ti.static(range(9)):
            _Feq[P][i] = w[i]*(_rou[P] + tg.scalar.smoothstep(_rou[P], 0, alpha)*(3 * e[i].dot(_FlowVelocity[P]) +
                               4.5 * (e[i].dot(_FlowVelocity[P]))**2 - 1.5 * _FlowVelocity[P].dot(_FlowVelocity[P])))


@ti.kernel
def fill_BG():
    for p in ti.grouped(_BackGroundLayer):
        _BackGroundLayer[p][0] = 1.0
        _BackGroundLayer[p][1] = 1.0
        _BackGroundLayer[p][2] = 1.0


# @ti.kernel
# def fill_PL(r: float, g: float, b: float, a: float):
#     for P in ti.grouped(_Pigment_Surface):
#         _Pigment_Surface[P][0] = r
#         _Pigment_Surface[P][1] = g
#         _Pigment_Surface[P][2] = b
#         _Pigment_Surface[P][3] = a


@ti.kernel
def render():
    for P in ti.grouped(_FrameBuffer):
        for i in ti.static(range(3)):
            # _FrameBuffer[P][i] = tg.scalar.mix(_BackGroundLayer[P][i],currentColor[0][i],_Pigment_Surface[P][3])
            _FrameBuffer[P][i] = tg.scalar.mix(
                _BackGroundLayer[P][i], currentColor[0][i], _Pigment_flow[P][3])
            # _FrameBuffer[P][i] = tg.scalar.mix(_FrameBuffer[P][i],currentColor[0][i],_Fixture[P][3])
            # _FrameBuffer[P][i] = tg.scalar.mix(_FrameBuffer[P][i],currentColor[1][i],_rou[P]*0.5)


@ti.kernel
def drawStrok(color: ti.template(), i: int, radius: ti.f32):
    center = ti.Vector([cursor[0], cursor[1]])
    for P in ti.grouped(_Pigment_Surface):
        dis = (P/res-center).norm()
        if dis < radius:
            # mask = max(1-_rou[P]/0.5,0.1)
            brush_tip = tg.scalar.clamp(1-dis/radius, 0, 1)
            _Water_Surface[P] += max(1-_rou[P]/0.5, 0.7)
            _Water_Surface[P] = tg.scalar.clamp(_Water_Surface[P])

            _Pigment_Surface[P][3] += max(1-_rou[P]/0.5, 0.7)*color[i][3]
            # _Pigment_Surface[P]=tg.scalar.clamp(_Pigment_Surface[P],color[i][3])
            _kar[P] = _paper[P]


psy = ti.field(dtype=float, shape=(res, res))


@ti.kernel
def waterSurface_to_flow():
    for P in ti.grouped(_Flow):
        psy[P] = tg.scalar.clamp(_Water_Surface[P], 0, 0.8-_rou[P])
        _Flow[P][0] += psy[P]
        _Water_Surface[P] -= psy[P]


@ti.kernel
def Pigment_S_to_F():
    for P in ti.grouped(psy):
        # if ( psy[P]>0):
        #     _Pigment_flow[P][3]=tg.scalar.clamp(_Pigment_flow[P][3]+_Pigment_Surface[P][3]*psy[P])
        #     _Pigment_Surface[P][3]=tg.scalar.clamp(_Pigment_Surface[P][3]-_Pigment_Surface[P][3]*psy[P])

        if (psy[P] > 0):
            denom = (_rou[P]+psy[P])
            _Pigment_flow[P] = (_Pigment_flow[P]*_rou[P] +
                                _Pigment_Surface[P]*psy[P])/denom
            # _Pigment_flow[P][3]=tg.scalar.clamp(_Pigment_flow[P][3],0,1)
            _Pigment_Surface[P][3] = tg.scalar.clamp(
                _Pigment_Surface[P][3]-(psy[P]/denom), 0, 1)


@ti.kernel
def update_Pf_star():
    for P in ti.grouped(_Pigment_flow_star):
        _Pigment_flow_star[P] = ti.Vector([0.0, 0.0, 0.0, 0.0])
        if (_Water_Surface[P] > 0):
            for i in ti.static(range(9)):
                _Pigment_flow_star[P] += _Flow[P][i] * \
                    _Pigment_flow[P-e[i]]
            _Pigment_flow_star[P] /= _rou[P]
        else:
            _Pigment_flow_star[P] = tg.sampling.bilerp(
                _Pigment_flow, P-4*_FlowVelocity[P])


@ti.kernel
def update_Pf():
    for P in ti.grouped(_Pigment_flow):
        gama_star = tg.scalar.mix(1, 0.1, tg.scalar.smoothstep(
            _FlowVelocity[P].norm()*4, 0, 0.05))
        _Pigment_flow[P] = (_Pigment_flow[P]-_Pigment_flow_star[P]
                            )*gama_star+_Pigment_flow_star[P]


_rouPre = ti.field(dtype=float, shape=(res, res))


@ti.kernel
def _update_rouPre():
    for P in ti.grouped(_rouPre):
        _rouPre[P] = _rou[P]


miu = -0.2
ksy = 0.5
niu = 0.0001


@ti.kernel
def update_Fixture():
    for P in ti.grouped(_Fixture):
        fixFactor = 0.0
        wLoss = max(_rouPre[P]-_rou[P], 0)
        if (wLoss > 0):
            fixFactor = wLoss/_rouPre[P]
        u_star = tg.scalar.clamp(miu+ksy*_Pigment_flow[P][3], 0, 1)
        fixFactor = max(
            fixFactor*(1-tg.scalar.smoothstep(_rou[P], 0, u_star)), niu)
        tempV = fixFactor*_Pigment_flow[P][3]
        # _Fixture[P]+=tempV
        _Fixture[P][3] = tg.scalar.clamp(_Fixture[P][3]+tempV)
        _Pigment_flow[P][3] = tg.scalar.clamp(_Pigment_flow[P][3]-tempV)


@ti.kernel
def ini_to_0(field: ti.template()):
    for P in ti.grouped(field):
        field[P] = ti.Vector([0, 0, 0, 0])


gui = ti.GUI("ti-ink", (res, res))
fill_BG()
ini_k()
ini_sigma()
cursor[0] = 0.5
cursor[1] = 0.5
ini_to_0(_Pigment_Surface)
ini_to_0(_Pigment_flow)
ini_to_0(_Fixture)
while gui.running:

    gui.get_event()

    if(gui.is_pressed(ti.GUI.LMB)):
        cursor[0] = gui.get_cursor_pos()[0]
        cursor[1] = gui.get_cursor_pos()[1]
        drawStrok(currentColor, 0, brushRadius)

    if(gui.is_pressed(ti.GUI.RMB)):
        cursor[0] = gui.get_cursor_pos()[0]
        cursor[1] = gui.get_cursor_pos()[1]
        drawStrok(currentColor, 1, brushRadius)
    waterSurface_to_flow()
    update_rou()
    update_Pf_star()
    update_Pf()
    # _update_rouPre()

    Pigment_S_to_F()
    get_ispinning()
    update_k_for_pinning()
    update_karpa()
    update_Feq()
    update_FlowNext()
    update_flow()

    # update_Fixture()
    render()
    gui.set_image(_FrameBuffer)
    gui.show()
