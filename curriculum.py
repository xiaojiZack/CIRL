class curriclum():
    def __init__(self) -> None:
        self.distance = [10000,10000,10000,10000,10000]
        self.velocity = [1000,1000,1000,1000,1000]
        self.chaos = [0,1,2,3,5]
        self.aScaleT = [9,9,9,9,9]
        self.weight4 = [1,1,0.1,0.01,0]
        self.delay = [0,1,5,10,20]
        self.max_stage_train = 2000
        self.threold_loss = 0.1
        self.stage = 0
        self.last_cnt = 0
    
    def is_meet(self,acc,loss,train_cnt) -> bool:
        if self.stage == 4: return False
        if train_cnt < 0.1*self.max_stage_train+self.last_cnt: return False
        if train_cnt>self.max_stage_train+self.last_cnt: return True
        if (acc>0.9):
            return True
        return False
    
    def get_stage(self):
        stage_inf = {
            'distance':self.distance[self.stage],
            'velocity':self.velocity[self.stage],
            'chaos':self.chaos[self.stage],
            'aScaleT':self.aScaleT[self.stage],
            'weight4':self.weight4[self.stage],
            'delay':self.delay[self.stage],
        }
        return stage_inf
    
    def nextstage(self,train_cnt):
        self.stage += 1
        self.last_cnt = train_cnt