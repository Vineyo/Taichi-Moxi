import taichi as ti
import matplotlib.image as mpig
import taichi_glsl as tg

ti.init(arch=ti.cuda)

@ti.data_oriented
class Taichi_Moxi:

    def __init__(self,res=512) -> None:
        self.RGB = ti.types.vector(3, float)
        self.RGBA = ti.types.struct(rgb=self.RGB, a=float)

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
        self.k = (0, 3, 4, 1, 2, 7, 8, 5, 6)
        self.w=ti.Vector([4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0 /
                    9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])
        self.res=res
        self.brushRadius = 0.03
        self.omiga = 0.5
        self.pigment_flow_rate=4.0
        self.q1=-0.005
        self.q2=0.9
        self.q3=0.9
        self.currntColor=ti.field(float,shape=4)
        self.cursor=ti.field(float,shape=2)
        self.Pigment_Surface_c = self.RGB.field(shape=(res, res))
        self.Pigment_Surface_a=ti.field(dtype=float,shape=(res,res))
        self.Pigment_flow_c = self.RGB.field(shape=(res, res))
        self.Pigment_flow_a=ti.field(dtype=float,shape=(res,res))
        self.Pigment_flow_star_c = self.RGB.field(shape=(res, res))
        self.Pigment_flow_star_a=ti.field(dtype=float,shape=(res,res))
        self.Water_Surface = ti.field(float, shape=(res, res))
        self.Flow = ti.Vector.field(9, dtype=float, shape=(res, res))
        self.FlowNext = ti.Vector.field(9, dtype=float, shape=(res, res))
        self.Feq = ti.Vector.field(9, dtype=float, shape=(res, res))
        self.kar_avg = ti.Vector.field(9, dtype=float, shape=(res, res))
        self.paper = ti.field(dtype=float, shape=(res, res))
        self.fibers = ti.field(dtype=float, shape=(res, res))
        self.rou = ti.field(float, shape=(res, res))
        self.sigma = ti.field(float, shape=(res, res))
        self.kar = ti.field(dtype=float, shape=(res, res))
        self.FlowVelocity = ti.Vector.field(2, dtype=float, shape=(res, res))
        self.psy = ti.field(dtype=float, shape=(res, res))
        self.FrameBuffer=ti.Vector.field(3, dtype=float, shape=(res, res))
        self.BackgrounLayer=ti.Vector.field(3, dtype=float, shape=(res, res))
        self.paper.from_numpy(mpig.imread("paper_512_2.png")[:, :, 0])
        self.fibers.from_numpy(mpig.imread("fibers_512_3.png")[:, :, 0])
        self.ini_k()
        self.update_kar_avg()
        self.set_edge_parameters()
        self.update_sigma()
        self.fill_BG()


    @ti.kernel
    def fill_BG(self):
        for p in ti.grouped(self.BackgrounLayer):
            self.BackgrounLayer[p] = ti.Vector([1,1,1])
            
    @ti.kernel
    def ini_k(self):
        for P in ti.grouped(self.kar):
            self.kar[P] = self.paper[P]
    @ti.kernel
    def update_kar_avg(self):
        for P in ti.grouped(self.kar_avg):
            for i in ti.static(range(9)):
                P_nebor = P+self.e[i]
                self.kar_avg[P][i] = 0.5*(self.kar[P]+self.kar[P_nebor])

    @ti.kernel
    def update_sigma(self):
        for P in ti.grouped(self.sigma):
            self.sigma[P] = self.q1+self.q2*self.fibers[P]+self.q3*self.paper[P]

    def set_edge_parameters(self,q1=-0.005,q2=0.9,q3=0.9):
        self.q1=q1
        self.q2=q2
        self.q3=q3

    @ti.kernel
    def update_k_for_pinning(self):
        for P in ti.grouped(self.kar):
            if (self.rou[P] == 0 and
                self.rou[P+self.e[1]] < self.sigma[P+self.e[1]] and
                self.rou[P+self.e[2]] < self.sigma[P+self.e[2]] and
                self.rou[P+self.e[3]] < self.sigma[P+self.e[3]] and
                self.rou[P+self.e[4]] < self.sigma[P+self.e[4]] and
                self.rou[P+self.e[5]] < (self.sigma[P+self.e[5]]*1.414213562373095048) and
                self.rou[P+self.e[6]] < (self.sigma[P+self.e[6]]*1.414213562373095048) and
                self.rou[P+self.e[7]] < (self.sigma[P+self.e[7]]*1.414213562373095048) and
                self.rou[P+self.e[8]] < (self.sigma[P+self.e[8]]*1.414213562373095048)):
                self.kar[P] = 1
            
    @ti.kernel
    def update_rou(self):
        for P in ti.grouped(self.rou):
            self.rou[P] = self.Flow[P].sum()*(0.995)

    @ti.kernel
    def update_FlowNext(self):
        for P in ti.grouped(self.FlowNext):
            for i in ti.static(range(9)):
                prePos = P-self.e[i]
                self.FlowNext[P][i] = self.omiga * \
                    (self.Feq[prePos][i]-self.Flow[prePos][i])+self.Feq[prePos][i]


    @ti.kernel
    def update_flow(self):
        for P in ti.grouped(self.Flow):
            # self.Flow[P]=self.FlowNext[P]
            for i in ti.static(range(9)):
                prePos = P-self.e[i]
                self.Flow[P][i] = self.kar_avg[P][i] * \
                    (self.FlowNext[P][self.k[i]] - self.FlowNext[prePos][i]) + \
                    self.FlowNext[prePos][i]

    @ti.kernel
    def update_Feq(self):
        alpha = 0.5
        for P in ti.grouped(self.Feq):
            self.FlowVelocity[P] = ti.Vector([0.0, 0.0])
            for j in ti.static(range(1, 9)):
                self.FlowVelocity[P] += self.Flow[P][j]*self.e[j]
            for i in ti.static(range(9)):
                self.Feq[P][i] = self.w[i]*(self.rou[P] + tg.scalar.smoothstep(self.rou[P], 0, alpha)*(3 * self.e[i].dot(self.FlowVelocity[P]) +
                                4.5 * (self.e[i].dot(self.FlowVelocity[P]))**2 - 1.5 * self.FlowVelocity[P].dot(self.FlowVelocity[P])))

    @ti.kernel
    def drawStrok(self):
        center = ti.Vector([self.cursor[0], self.cursor[1]])
        for P in ti.grouped(self.Pigment_Surface_c):
            dis = (P/self.res-center).norm()
            if dis < self.brushRadius:
                brush_tip = tg.scalar.clamp(1-dis/self.brushRadius, 0, 1)
                self.Water_Surface[P] += max(1-self.rou[P]/0.5, 0.7)
                self.Water_Surface[P] = tg.scalar.clamp(self.Water_Surface[P])
                self.Pigment_Surface_c[P] = ti.Vector([self.currntColor[0],self.currntColor[1],self.currntColor[2]])
                self.Pigment_Surface_a[P] += brush_tip*self.currntColor[3]
                self.kar[P] = self.paper[P]

    @ti.kernel
    def waterSurface_to_flow(self):
        for P in ti.grouped(self.Flow):
            self.psy[P] = tg.scalar.clamp(self.Water_Surface[P], 0, 1.0-self.rou[P])
            self.Flow[P][0] += self.psy[P]
            self.Water_Surface[P] -= self.psy[P]

    @ti.kernel
    def Pigment_S_to_F(self):
        for P in ti.grouped(self.psy):
            if (self.psy[P] > 0):
                denom = (self.rou[P]+self.psy[P])
                self.Pigment_flow_c[P] = (self.Pigment_flow_c[P]*self.rou[P]*self.Pigment_flow_a[P] +
                                        self.Pigment_Surface_c[P]*self.psy[P]*self.Pigment_Surface_a[P])/  \
                                        (self.rou[P]*self.Pigment_flow_a[P]+self.psy[P]*self.Pigment_Surface_a[P])
                self.Pigment_flow_a[P] = tg.scalar.clamp(
                    self.Pigment_flow_a[P]+self.psy[P]*self.Pigment_Surface_a[P]/denom, 0, 1)
                self.Pigment_Surface_a[P] = tg.scalar.clamp(
                    self.Pigment_Surface_a[P]-(self.psy[P]/denom), 0, 1)

    @ti.kernel
    def update_Pf_star(self):
        for P in ti.grouped(self.Pigment_flow_star_c):
            self.Pigment_flow_star_c[P]=tg.sampling.bilerp(self.Pigment_flow_c,P-self.pigment_flow_rate*self.FlowVelocity[P])
            self.Pigment_flow_star_a[P] = tg.sampling.bilerp(self.Pigment_flow_a, P-self.pigment_flow_rate*self.FlowVelocity[P])

    @ti.kernel
    def update_Pf(self):
        for P in ti.grouped(self.Pigment_flow_c):
            gama_star = tg.scalar.mix(1, 0.1, tg.scalar.smoothstep(
                self.FlowVelocity[P].norm()*self.pigment_flow_rate, 0, 0.05))
            self.Pigment_flow_a[P] = (self.Pigment_flow_a[P]-self.Pigment_flow_star_a[P]
                                )*gama_star+self.Pigment_flow_star_a[P]
            self.Pigment_flow_c[P] = (self.Pigment_flow_c[P]-self.Pigment_flow_star_c[P]
                            )*gama_star+self.Pigment_flow_star_c[P]
            # self.Pigment_flow_c[P] = (self.Pigment_flow_c[P]-self.Pigment_flow_star_c[P]
            #                     )*self.Pigment_flow_a[P]/(self.Pigment_flow_a[P]+self.Pigment_flow_star_a[P])+self.Pigment_flow_star_c[P]

    @ti.kernel
    def render(self):
        for P in ti.grouped(self.FrameBuffer):
            self.FrameBuffer[P] = tg.scalar.mix(
                self.BackgrounLayer[P], self.Pigment_Surface_c[P], self.Pigment_Surface_a[P])
            self.FrameBuffer[P] = tg.scalar.mix(
                self.FrameBuffer[P], self.Pigment_flow_c[P], self.Pigment_flow_a[P])
            self.FrameBuffer[P] = tg.scalar.mix(
                self.FrameBuffer[P], ti.Vector([0.0, 0.2, 0.6]), self.rou[P]*0.5)


    def update(self):
        self.waterSurface_to_flow()
        self.Pigment_S_to_F()
        self.update_rou()
        self.update_Feq()
        self.update_FlowNext()
        self.update_flow()
        self.update_Pf_star()
        self.update_Pf()
        self.update_k_for_pinning()
        self.update_kar_avg()

    def setCurrentColor(self,r=0.0,g=0.0,b=0.0,a=0.0):
        self.currntColor[0]=r
        self.currntColor[1]=g
        self.currntColor[2]=b
        self.currntColor[3]=a
    def setBrushRadius(self,radius):
        self.brushRadius=radius
   
    def setCursor(self,x,y):
        self.cursor[0]=x
        self.cursor[1]=y