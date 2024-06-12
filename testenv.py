"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import spaces

class OneVOneEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    1v1导弹拦截场景，场景采用博士论文的内容。不考虑目标自主智能机动

    ### Action Space

    输入3个值，一个是俯仰角总过载N，一个是俯仰角过载Nz/偏航角过载Ny范围，偏航角过载Ny范围[-1,1]

    ### Observation Space

    //返回给神经网络的观测部分
    距离D
    相对速度D_dot
    弹目俯仰角qe
    弹目偏航角qb
    弹目俯视角速率qe_dot
    弹目偏航角速率qb_dot
    自身速率vm
    //不返回的部分
    追者坐标[xm,ym,zm]
    追者速度/俯仰角/偏航角[vm,setam,puxim]
    逃者坐标[xt,yt,zt]
    逃者速度/俯仰角/偏航角[vt,setat,puxit]
    逃者机动[My,Mz]
    """

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.low_height = 0
        self.high_height = 30000
        self.fast_velocity = 3*340
        self.gap = 0.01     #仿真步长 s
        self.aScaleM = 40   #导弹机动能力 g
        self.aScaleT = 9    #目标机动能力
        self.maxstep = 10000  #最大环境步数
        
        self.terminalDist = 10  #终止距离m
        self.dm = 4000          #机动距离m
        self.terminalEnd = 100  #击中角度，未实装
        self.terminalVm =500   #终止速度m/s
    
        self.wd=0.01   #密集奖励权重
        self.fd=10      #命中奖励权重
        
        self.nearstD = 10000  #起始距离   
        
        self.chaos = 0

        high = np.array(
            [
                50000,
                self.fast_velocity,
                90,
                360,
                90, 180,
                1
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                0,
                0,
                -90,
                0,
                -90, -180,
                0
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(low=-1,high=1,shape=[2,])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.render_mode = render_mode

        self.state = None   #环境当前状态
        self.view = None    #环境输出观测向量
        self.oldD = 0       #上一回合的相对距离
        self.oldView = None #环境上一回合的观测向量

        self.steps_beyond_terminated = None

    def step(self, action):
        """
        环境根据action给出reward和观测向量并进入下一状态
        """
        #assert self.action_space.contains(action), "动作超出动作空间"
        vm, xm, ym, zm, setam, puxim, vt, xt,yt,zt, setat, puxit, ty, tz = self.state
        Ny,Nz = action

        #计算具体过载量
        Ny = np.clip(Ny,-1,1)
        Nz = np.clip(Nz,-1,1)
        aNy = Ny*self.aScaleM
        aNz = Nz*self.aScaleM
        aty = ty*self.aScaleT
        atz = tz*self.aScaleT

        #计算导弹/目标的俯仰角/偏航角变化率
        setam_dot = self.gravity*(aNy/vm)
        setat_dot = self.gravity*(aty/vt)
        puxim_dot = self.gravity*(aNz/(vm*math.cos(setam)))
        puxit_dot = self.gravity*(atz/(vt*math.cos(setat)))

        #计算相对坐标系的速度
        xm_dot = vm*math.cos(setam)*math.cos(puxim)
        ym_dot = vm*math.sin(setam)
        zm_dot = vm*math.cos(setam)*math.sin(puxim)
        xt_dot = vt*math.cos(setat)*math.cos(puxit)
        yt_dot = vt*math.sin(setat)
        zt_dot = vt*math.cos(setat)*math.sin(puxit)

        #计算俯仰角/偏航角变化
        newSetam = setam+setam_dot*self.gap
        newPuxim = puxim+puxim_dot*self.gap
        if (newSetam<-math.pi*0.5):             #当俯仰角大于90或小于-90时，进行修正
            newSetam = -math.pi-newSetam
            newPuxim = math.pi+newPuxim
        if (newSetam>math.pi*0.5): 
            newSetam = math.pi-newSetam
            newPuxim = math.pi+newPuxim   
        newPuxim = (2*math.pi+newPuxim)%(2*math.pi)

        #坐标位置更新
        newXm = xm+xm_dot*self.gap
        newYm = ym+ym_dot*self.gap
        newZm = zm+zm_dot*self.gap
        newVm = self.velocity_change(xm,ym,zm,vm,setam) #导弹下一回合速度更新
        newSetat = (setat+setat_dot*self.gap)
        newPuxit = puxit+puxit_dot*self.gap
        newPuxit = (2*math.pi+newPuxit)%(2*math.pi)
        if (newSetat<-math.pi*0.5): newSetat = -math.pi*0.5
        if (newSetat>math.pi*0.5): newSetat = math.pi*0.5  
        newXt = xt+xt_dot*self.gap
        newYt = yt+yt_dot*self.gap
        newZt = zt+zt_dot*self.gap
        newVt = vt
        
        #相对距离，相对速度向量计算
        coorM = np.array((newXm, newYm, newZm))
        coorT = np.array((newXt, newYt, newZt))
        coorM_dot = np.array((xm_dot, ym_dot, zm_dot))
        coorT_dot = np.array((xt_dot, yt_dot, zt_dot))
        xr, yr, zr = coorT-coorM
        xr_dot, yr_dot, zr_dot = coorT_dot-coorM_dot

        d = np.linalg.norm(coorM-coorT)
        d_dot = (xr*xr_dot+yr*yr_dot+zr*zr_dot)/d
        qe = math.atan2(yr,math.sqrt(xr*xr+zr*zr))      #新回合视线俯仰角计算
        qb = math.atan2(zr,xr)                          #新回合视线偏航角计算
        qe_dot = 1/(d**2)*(yr_dot*math.sqrt(xr**2+zr**2)-(xr*xr_dot+zr*zr_dot)/math.sqrt(xr**2+zr**2)*yr)#(yr_dot*(xr**2+zr**2)-(xr*xr_dot+zr*zr_dot)*yr)/(math.sqrt(xr**2+zr**2)*(d**2))   #视线俯仰角变化率计算
        qb_dot = (zr_dot*xr-xr_dot*zr)/(xr*xr+zr*zr)                                                        #视线偏航角变化率计算
        

        meet_angle = math.acos(round(np.sum(coorM_dot*coorT_dot)/(np.linalg.norm(coorM_dot)*np.linalg.norm(coorT_dot)),4))  #交汇角计算


        #当进入机动距离后，目标生成机动过载
        if (d<self.dm and ty ==0 and tz == 0):
            ty = np.random.random()*2-1
            tz = math.sqrt(1-ty*ty)*np.sign(np.random.randint(2)*2-1)

        #更新环境状态
        self.state = (
            newVm, newXm, newYm, newZm, newSetam, newPuxim,
            newVt, newXt, newYt, newZt, newSetat, newPuxit,
            ty, tz
            )

        #检查终止条件
        terminated = bool(d<=self.terminalDist or newVm<self.terminalVm or self.epstep>self.maxstep)
        if (d<=self.terminalDist and meet_angle>(105/180*math.pi)):
            true_hit = True
        else: true_hit = False
        
        #奖励函数
        reward = self.wd*(self.oldD-d-1*math.cos(meet_angle)) #稀疏奖励
        if (d<=self.terminalDist): reward =reward +self.fd*(1-1*math.cos(meet_angle)) #命中奖励



        ############
        self.oldView2 = self.oldView
        self.oldView = self.view  #保存上一回合观测状态
        self.oldD = d
        d = np.clip(d,0,5000)/5000                  #相对距离截断
        d_dot = np.clip(d_dot,-1000,1000)/1000      #相对速度截断
        # ############# 加干扰
        # chaos_qe = self.np_random.normal(qe,self.chaos/100*abs(qe))
        # chaos_qb = self.np_random.normal(qb,self.chaos/100*abs(qb))
        # chaos_qe_dot = self.np_random.normal(qe_dot,self.chaos/100*abs(qe_dot))
        # chaos_qb_dot = self.np_random.normal(qe_dot,self.chaos/100*abs(qb_dot))
        chaos_d = self.np_random.normal(d,self.chaos/100*abs(d)+0.0002)
        chaos_d_dot = self.np_random.normal(d_dot,self.chaos/100*abs(d_dot)+0.001)
        # #############
        #self.view = (d, d_dot,newVm/1000,chaos_qe/math.pi, chaos_qb/math.pi, newSetam/math.pi, chaos_qe_dot, chaos_qb_dot)  #观测向量生成
        #self.view = (d, d_dot,newVm/1000,qe/math.pi, qb/math.pi, newSetam/math.pi, qe_dot, qb_dot)  #观测向量生成
        self.view = (chaos_d, chaos_d_dot,newVm/1000,qe/math.pi, qb/math.pi, newSetam/math.pi, qe_dot, qb_dot)
        self.epstep = self.epstep+1
        
        return np.array((self.view+self.oldView2), dtype=np.float32).reshape(16), reward, terminated, d, true_hit, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        stage_inf = None
    ):
        """
            重置环境
        """
        super().reset(seed=seed)
        self.epstep = 0
        vm = 1000.0
        xm, ym, zm = (0.0,0.0,0.0)
        setam = 0.25*math.pi#+self.np_random.uniform(-10.0,10.0)/180*math.pi        #导弹速度俯仰角
        puxim = 0.25*math.pi#self.np_random.uniform(0.0,360.0)/180.0*math.pi#math.pi             #导弹速度偏航角
        vt = 300                                                                    #目标速度
        start_d = 10000                                                              #起始距离
        
        if stage_inf:
            start_d = stage_inf['distance']
            vm = stage_inf['velocity']
            #self.aScaleT = self.np_random.uniform(max(0, stage_inf['aScaleT']-4),stage_inf['aScaleT'])
            self.aScaleT = stage_inf['aScaleT']
            self.chaos = stage_inf['chaos']
            
        yt =2500.0#start_d*math.sin(setam)#self.np_random.uniform(1000,3800)               #目标初始化位置坐标
        xt = 4500.0#math.sqrt(start_d**2-yt*yt)*math.cos(puxim)
        zt = 4500.0#math.sqrt(start_d**2-yt*yt)*math.sin(puxim)
        setat = 0#-0.25*math.pi#+self.np_random.uniform(-20.0,10.0)/180*math.pi       #目标初始化速度俯仰角
        puxit = -math.pi + puxim#+self.np_random.uniform(-15.0,15.0)/180*math.pi    #目标初始化速度偏航角
        self.state = (
            vm, xm, ym, zm, setam, puxim, vt, xt,yt,zt, setat, puxit, 0, 0
        )
        self.nearstD = 5000
        
        xm_dot = vm*math.cos(setam)*math.cos(puxim)
        ym_dot = vm*math.sin(setam)
        zm_dot = vm*math.cos(setam)*math.sin(puxim)
        xt_dot = vt*math.cos(setat)*math.cos(puxit)
        yt_dot = vt*math.sin(setat)
        zt_dot = vt*math.cos(setat)*math.sin(puxit)

        coorM = np.array((xm, ym, zm))
        coorT = np.array((xt, yt, zt))
        coorM_dot = np.array((xm_dot, ym_dot, zm_dot))
        coorT_dot = np.array((xt_dot, yt_dot, zt_dot))
        xr, yr, zr = coorT-coorM
        xr_dot, yr_dot, zr_dot = coorT_dot-coorM_dot

        d = np.linalg.norm(coorM-coorT)
        self.oldD = d
        d_dot = (xr*xr_dot+yr*yr_dot+zr*zr_dot)/d
        qe = math.atan2(yr,math.sqrt(xr*xr+zr*zr))
        qb = math.atan2(zr,xr)
        #qe_dot = ((xr*xr)+zr*zr)*yr_dot-(yr*(xr*xr_dot)/math.sqrt(xr*xr+zr*zr))
        qe_dot = 1/(d**2)*(yr_dot*math.sqrt(xr**2+zr**2)-(xr*xr_dot+zr*zr_dot)/math.sqrt(xr**2+zr**2)*yr)
        qb_dot = (zr_dot*xr-xr_dot*zr)/(xr*xr+zr*zr)

        d = np.clip(d,0,5000)/5000
        d_dot = np.clip(d_dot,-1000,1000)/1000
        self.view = (d, d_dot,1,qe/math.pi,qb/math.pi,setam/math.pi, qe_dot, qb_dot)
        self.oldView = self.view
        self.steps_beyond_terminated = None

        self.dm = self.np_random.uniform(4000.0,6000.0)     

        return np.array((self.view,self.view), dtype=np.float32).reshape(16), d

    def render(self):
        pass
    
    def velocity_change(self,x,y,z,v,seta):
        '''
        计算追者经过时间t后的速度
        '''
        leta1 = -1.15e-4
        leta2 = -1.62e-4
        Sa = 0.1
        Hdk = y+(x*x+y*y+z*z)/12756490
        if (Hdk<=1.1e4):
            Hq = 0.06*math.exp(leta1*Hdk)*v*v
        else:
            Hq = 0.01*math.exp(leta2*(Hdk-1.1e4))*v*v
        #if (-1*(self.cal_Ca(Hdk,v)*Hq*Sa)/600-math.sin(seta))>0: print(f"速度增大{Hdk},{v},{Hq},{seta},{self.cal_Ca(Hdk,v)},{math.sin(seta)},{(-1*(self.cal_Ca(Hdk,v)*Hq*Sa)/600-math.sin(seta))}")
        return v+self.gap*self.gravity*(-1*(self.cal_Ca(Hdk,v)*Hq*Sa)/600-math.sin(seta))

    def cal_Ca(self,h,v):
        '''
        计算空气动力学系数
        采用二次线性插值法
        '''
        def find_tmp(x,y,weight,total):
            return x+(y-x)*(weight/total)
        tableV = [0.1,0.3,0.6,0.8,0.9,1.0,1.1,1.2,1.5,2.0,3.0] #单位Ma
        tableH = [0,15000,30000] #单位m
        tableV = [i * 340.3 for i in tableV] #单位m/s
        table = [
            [0.403, 0.407, 0.412, 0.421, 0.445, 0.473, 0.478, 0.481, 0.399, 0.311, 0.228],
            [0.444, 0.444, 0.456, 0.478, 0.503, 0.522, 0.523, 0.433, 0.350, 0.266, 0.214],
            [0.563, 0.555, 0.545, 0.568, 0.568, 0.591, 0.587, 0.509, 0.408, 0.311, 0.254]
        ]

        maxV = 0
        minV = 0
        while(tableV[maxV]<v): maxV+=1
        maxH = 0
        minH = 0
        while(tableH[maxH]<h): maxH+=1
        minV = maxV-1
        minH = maxH-1

        tmp1 = find_tmp(table[minH][minV],table[minH][maxV],v-tableV[minV],tableV[maxV]-tableV[minV])
        tmp2 = find_tmp(table[maxH][minV],table[maxH][maxV],v-tableV[minV],tableV[maxV]-tableV[minV])

        return find_tmp(tmp1,tmp2, h-tableH[minH], tableH[maxH]-tableH[minH])