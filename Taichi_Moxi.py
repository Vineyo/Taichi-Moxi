from numpy import dtype, int32
import taichi as ti
import matplotlib.image as mpig


@ti.data_oriented
class Taichi_Moxi:

    def __init__(self, res=512) -> None:

        self.k = (0, 3, 4, 1, 2, 7, 8, 5, 6)
        self.res = res
        self.alpha = 0.2
        self.omiga = 0.1
        self.q = ti.field(float, shape=3)
        self.pigment_flow_rate = ti.field(float, shape=())
        self.paper_dryness = ti.field(float, shape=())
        self.currntColor = ti.field(float, shape=4)
        self.activated_gravity = ti.field(ti.i32, shape=())
        self.cursor = ti.field(float, shape=2)
        self.Pigment_Surface_c = ti.Vector.field(
            3, dtype=float, shape=(res, res))
        self.Pigment_Surface_a = ti.field(dtype=float, shape=(res, res))
        self.FrameBuffer = ti.Vector.field(3, dtype=float, shape=(res, res))
        self.BackgrounLayer = ti.Vector.field(3, dtype=float, shape=(res, res))
        self.Water_Surface = ti.field(float, shape=(res, res))
        self.kappa_avg = ti.Vector.field(9, dtype=float, shape=(res, res))
        self.sigma = ti.field(float, shape=(res, res))
        self.kappa = ti.field(dtype=float, shape=(res, res))
        self.paper = ti.field(dtype=float, shape=(res, res))
        self.fibers = ti.field(dtype=float, shape=(res, res))
        self.w = ti.Vector.field(9, dtype=float, shape=2)
        self.Pigment_flow_c = ti.Vector.field(3, dtype=float)
        self.Pigment_flow_a = ti.field(dtype=float)
        self.Pigment_flow_star_c = ti.Vector.field(3, dtype=float)
        self.Pigment_flow_star_a = ti.field(dtype=float)
        self.Flow = ti.Vector.field(9, dtype=float)
        self.FlowNext = ti.Vector.field(9, dtype=float)
        self.Feq = ti.Vector.field(9, dtype=float)
        self.rho = ti.field(float)
        self.FlowVelocity = ti.Vector.field(2, dtype=float)

        self.s0 = ti.root
        self.s1 = self.s0.dense(ti.ij, int(self.res/32))
        self.s2 = self.s1.dense(ti.ij, 8)
        self.s3 = self.s2.dense(ti.ij, 4)

        self.s3.place(self.Pigment_flow_c, self.Pigment_flow_a)
        self.s3.place(self.Pigment_flow_star_c, self.Pigment_flow_star_a)
        self.s3.place(self.Flow, self.Feq, self.FlowNext)
        self.s3.place(self.rho)
        self.s3.place(self.FlowVelocity)

        self.w[0] = ti.Vector([4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0 /
                               9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])
        self.w[1] = ti.Vector([4.0/9.0, 0.95/9.0, 1.0/9.0, 1.05/9.0, 1.0 /
                               9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])

        self.e = ti.Vector.field(2, dtype=int, shape=9)
        self.e[0] = ti.Vector([0, 0])
        self.e[1] = ti.Vector([0, 1])
        self.e[2] = ti.Vector([-1, 0])
        self.e[3] = ti.Vector([0, -1])
        self.e[4] = ti.Vector([1, 0])
        self.e[5] = ti.Vector([1, 1])
        self.e[6] = ti.Vector([-1, 1])
        self.e[7] = ti.Vector([-1, -1])
        self.e[8] = ti.Vector([1, -1])
        self._paper = ti.field(float, shape=(512, 512))
        self._paper.from_numpy(mpig.imread("paper_512_2.png")[:, :, 0])
        self._fibers = ti.field(float, shape=(512, 512))
        self._fibers.from_numpy(mpig.imread("fibers_512_2.png")[:, :, 0])
        self.init_paper_and_fibers()
        self.pigment_flow_rate[None] = 4
        self.paper_dryness[None] = 0.5
        self.activated_gravity[None] = 0
        self.Pigment_flow_c.fill(1)
        self.ini_k()
        self.update_kappa_avg()
        self.set_edge_parameters(q1=0.01, q2=.07, q3=0.2)
        self.update_sigma()
        self.BackgrounLayer.fill(1.0)

    @ti.kernel
    def init_paper_and_fibers(self):
        for i, j in ti.ndrange(self.res, self.res):
            self.paper[i, j] = self._paper[i % 512, j % 512]
            self.fibers[i, j] = self._fibers[i % 512, j % 512]

    @ti.kernel
    def ini_k(self):
        for P in ti.grouped(self.kappa):
            self.kappa[P] = self.paper[P]

    @ti.kernel
    def update_kappa_avg(self):
        for P in ti.grouped(self.kappa_avg):
            for i in ti.static(range(1, 9)):
                self.kappa_avg[P][i] = 0.5 * \
                    (self.kappa[P]+self.kappa[P+self.e[i]])

    @ti.kernel
    def update_sigma(self):
        for P in ti.grouped(self.sigma):
            self.sigma[P] = self.q[0]+self.q[1] * \
                self.fibers[P]+self.q[2]*self.paper[P]

    @ti.kernel
    def update_k_for_pinning(self):
        for P in ti.grouped(self.rho):
            if (self.rho[P] < 1e-6 and
                self.rho[P+self.e[1]] < self.sigma[P+self.e[1]] and
                self.rho[P+self.e[2]] < self.sigma[P+self.e[2]] and
                self.rho[P+self.e[3]] < self.sigma[P+self.e[3]] and
                self.rho[P+self.e[4]] < self.sigma[P+self.e[4]] and
                self.rho[P+self.e[5]] < (self.sigma[P+self.e[5]]*1.41421356) and
                self.rho[P+self.e[6]] < (self.sigma[P+self.e[6]]*1.41421356) and
                self.rho[P+self.e[7]] < (self.sigma[P+self.e[7]]*1.41421356) and
                    self.rho[P+self.e[8]] < (self.sigma[P+self.e[8]]*1.41421356)):
                self.kappa[P] = self.paper_dryness[None]

    @ti.kernel
    def update_rho(self):
        for P in ti.grouped(self.Flow):
            self.rho[P] = self.Flow[P].sum()*0.995

    @ti.kernel
    def update_FlowNext(self):
        for P in ti.grouped(self.Flow):
            for i in ti.static(range(9)):
                prePos = clamp(P-self.e[i], 0, self.res-1)
                self.FlowNext[P][i] = mix(
                    self.Feq[prePos][i], self.Flow[prePos][i], self.omiga)

    @ti.kernel
    def update_flow(self):
        for P in ti.grouped(self.FlowNext):
            for i in ti.static(range(9)):
                prePos = clamp(P-self.e[i], 0, self.res-1)
                self.Flow[P][i] = self.kappa_avg[P][i] * \
                    (self.FlowNext[P][self.k[i]] - self.FlowNext[prePos][i]) + \
                    self.FlowNext[prePos][i]

    @ti.kernel
    def update_Feq(self):

        for P in ti.grouped(self.rho):
            if self.rho[P] > 1e-6:
                self.FlowVelocity[P] = self.Flow[P][1]*self.e[1] +\
                    self.Flow[P][2]*self.e[2] +\
                    self.Flow[P][3]*self.e[3] +\
                    self.Flow[P][4]*self.e[4] +\
                    self.Flow[P][5]*self.e[5] +\
                    self.Flow[P][6]*self.e[6] +\
                    self.Flow[P][7]*self.e[7] +\
                    self.Flow[P][8]*self.e[8]

                for i in ti.static(range(9)):
                    self.Feq[P][i] = self.w[self.activated_gravity[None]][i]*(self.rho[P] + smoothstep(self.rho[P], 0, self.alpha)*(3 * self.e[i].dot(self.FlowVelocity[P]) +
                                                                                                                                    4.5 * (self.e[i].dot(self.FlowVelocity[P]))**2 - 1.5 * self.FlowVelocity[P].dot(self.FlowVelocity[P])))

    @ti.kernel
    def drawStrok(self, brushRadius: float, brush_wetness: float, softness: float):
        center = ti.Vector([self.cursor[0], self.cursor[1]])
        r = brushRadius**2
        for i, j in ti.ndrange((int(self.res*(center[0] - r)), int(self.res*(center[0] + r))), (int(self.res*(center[1] - r)), int(self.res*(center[1] + r)))):
            dis = (ti.Vector([i, j])/self.res-center).norm()
            if dis < r:
                brush_tip = smoothstep(0, softness, 1-dis/r)
                water_mask = max(1-self.rho[i, j]/0.5, 0.1)
                self.Water_Surface[i, j] += water_mask
                self.Water_Surface[i, j] = clamp(
                    self.Water_Surface[i, j], 0, 2*brush_wetness)
                self.Pigment_Surface_c[i, j] = ti.Vector(
                    [self.currntColor[0], self.currntColor[1], self.currntColor[2]])
                self.Pigment_Surface_a[i, j] += brush_tip*self.currntColor[3]
                self.Pigment_Surface_a[i, j] = clamp(
                    self.Pigment_Surface_a[i, j])
                self.kappa[i, j] = self.paper[i, j]

    @ti.kernel
    def water_pigment_surface_to_flow(self):
        for P in ti.grouped(self.Water_Surface):
            if self.Water_Surface[P] > 0:
                phi = clamp(
                    self.Water_Surface[P], 0, 1.0-self.rho[P])
                self.Flow[P][0] += phi

                b = phi/self.Water_Surface[P]*self.Pigment_Surface_a[P]
                self.Water_Surface[P] -= phi
                if b > 0:
                    self.Pigment_flow_c[P] = (self.Pigment_flow_c[P]*self.Pigment_flow_a[P] +
                                              self.Pigment_Surface_c[P]*b) / (self.Pigment_flow_a[P]+b)

                    self.Pigment_flow_a[P] = clamp(
                        self.Pigment_flow_a[P]+b, 0, 1)
                    self.Pigment_Surface_a[P] = clamp(
                        self.Pigment_Surface_a[P]-b, 0, 1)

    @ti.kernel
    def update_Pf_star(self):
        for P in ti.grouped(self.Pigment_flow_c):
            Ppre = P-self.pigment_flow_rate[None]*self.FlowVelocity[P]
            self.Pigment_flow_star_c[P] = bilerp(
                self.Pigment_flow_c, Ppre)
            self.Pigment_flow_star_a[P] = bilerp(
                self.Pigment_flow_a, Ppre)

    @ti.kernel
    def update_Pf(self):
        for P in ti.grouped(self.Pigment_flow_c):
            gama_star = mix(1, 0.005, smoothstep(
                self.FlowVelocity[P].norm()*self.pigment_flow_rate[None], 0, 0.001))
            self.Pigment_flow_a[P] = (self.Pigment_flow_a[P]-self.Pigment_flow_star_a[P]
                                      )*gama_star+self.Pigment_flow_star_a[P]
            self.Pigment_flow_c[P] = (self.Pigment_flow_c[P]-self.Pigment_flow_star_c[P]
                                      )*gama_star+self.Pigment_flow_star_c[P]

    @ti.kernel
    def render(self):
        for i, j in ti.ndrange(self.res, self.res):
            # s_index = ti.rescale_index(self.Flow, self.s1, [i, j])
            # if (ti.is_active(self.s1, s_index)):
            self.FrameBuffer[i, j] = mix(
                self.BackgrounLayer[i, j], self.Pigment_flow_c[i, j], self.Pigment_flow_a[i, j])
            self.FrameBuffer[i, j] = mix(
                self.FrameBuffer[i, j], self.Pigment_Surface_c[i, j], self.Pigment_Surface_a[i, j])
            # self.FrameBuffer[i, j] = mix(
            #     self.FrameBuffer[i, j], ti.Vector([0.0, 0.2, 0.6]), self.rho[i, j]*0.5)

    def update(self):

        self.water_pigment_surface_to_flow()
        self.update_rho()
        self.update_Feq()
        self.update_FlowNext()
        self.update_flow()
        self.update_Pf_star()
        self.update_Pf()
        self.update_k_for_pinning()
        self.update_kappa_avg()

    def setCurrentColor(self, r=0.0, g=0.0, b=0.0, a=0.0):
        self.currntColor[0] = r
        self.currntColor[1] = g
        self.currntColor[2] = b
        self.currntColor[3] = a

    def setCursor(self, x, y):
        self.cursor[0] = x
        self.cursor[1] = y

    def set_edge_parameters(self, q1=-0.05, q2=0.9, q3=0.9):
        self.q[0] = q1
        self.q[1] = q2
        self.q[2] = q3

    def clear(self):
        self.Pigment_flow_a.fill(0)
        self.Pigment_flow_c.fill(1)
        self.Pigment_Surface_a.fill(0)
        self.Pigment_Surface_c.fill(0)
        self.dry()

    def dry(self):
        self.Water_Surface.fill(0)
        self.Flow.fill(0)

    def save(self, filename):
        ti.imwrite(self.FrameBuffer.to_numpy(), filename)


@ti.func
def bilerp(filed, pos):
    "bilinear insertion"
    x, y = pos[0], pos[1]
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1
    x = x - x0
    y = y - y0
    return mix(mix(filed[x0, y0], filed[x1, y0], x),
               mix(filed[x0, y1], filed[x1, y1], x), y)


@ti.func
def clamp(x, a=0.0, b=1.0):
    return ti.max(a, ti.min(b, x))


@ti.func
def smoothstep(edge0, edge1, x):
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def mix(a, b, x):
    return a * (1.0 - x) + b * x
