
import math
import matplotlib
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import random
from collections import deque
from tqdm import tqdm
import typing as typ

from testenv import OneVOneEnv
from curriculum import curriclum

import shutil

from torch.utils.tensorboard import SummaryWriter

class policyNet(nn.Module):
    """
    Actor网络
    """
    def __init__(self, state_dim: int, hidden_layers_dim: typ.List, action_dim: int):
        super(policyNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_action': nn.ReLU(inplace=True)
            }))

        self.fc_mu = nn.Linear(hidden_layers_dim[-1], action_dim)
        self.fc_std = nn.Linear(hidden_layers_dim[-1], action_dim)

    def forward(self, x):
        for layer in self.features:
            x = layer['linear_action'](layer['linear'](x))
        
        mean_ = 1.0 * torch.tanh(self.fc_mu(x).float()) #动作均值
        # np.log(1 + np.exp(2))
        std = 0.5*torch.tanh(self.fc_std(x))+0.6      #动作方差
        # std = F.softplus(self.fc_std(x))+1e-8
        return mean_, std


class valueNet(nn.Module):
    """
    crtic网络
    """
    def __init__(self, state_dim, hidden_layers_dim):
        super(valueNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                'linear_activation': nn.ReLU(inplace=True)
            }))
        
        self.head = nn.Linear(hidden_layers_dim[-1] , 1)
        
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_activation'](layer['linear'](x))
        return self.head(x)


def compute_advantage(gamma, lmbda, td_delta, done):
    td_delta = td_delta.detach().numpy()
    adv_list = []
    adv = 0
    count = 1
    for delta in td_delta[::-1]:
        adv = gamma * lmbda * adv + delta
        adv_list.append(adv)
        count += 1
        count = min(count, len(td_delta))
        if (done[-count]>0): 
            adv = 0
    adv_list.reverse()
    return torch.FloatTensor(adv_list)


class PPO:
    """
    PPO算法, 主过程
    """
    def __init__(self,
                state_dim: int,
                hidden_layers_dim: typ.List,
                action_dim: int,
                actor_lr: float,
                critic_lr: float,
                gamma: float,
                PPO_kwargs: typ.Dict,
                device: torch.device,
                max_norm: float,
                weight2: float,
                weight3: float,
                weight4: float,
                ):
        self.actor = policyNet(state_dim, hidden_layers_dim, action_dim).to(device)
        self.critic = valueNet(state_dim, hidden_layers_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.lmbda = PPO_kwargs['lmbda']
        self.ppo_epochs = PPO_kwargs['ppo_epochs'] # 一条序列的数据用来训练的轮次
        self.eps = PPO_kwargs['eps'] # PPO中截断范围的参数
        self.count = 0 
        self.device = device
        self.max_norm = max_norm
        self.weight2 = weight2
        self.weight3 = weight3
        self.weight4 = weight4

    def get_rule(self, state):
        ruleaction = []
        for i in range(0,len(state)):
            s = state[i].tolist()
            v = s[2]*1000
            seta = s[5]*math.pi
            qe_dot = s[6]
            qb_dot = s[7]
            Ny = qe_dot*v/(9.8*40)#float(np.clip(qe_dot*v/(9.8*40),-1.0,1.0))
            Nz = qb_dot*v*math.cos(seta)/(9.8*40)#float(np.clip(qb_dot*v*math.cos(seta)/(9.8*40),-1.0,1.0))
            ruleaction.append([Ny*3,Nz*3])
        return torch.FloatTensor(ruleaction).to(self.device)
    
    def huber_lose(self, a, b):
        hb = 1
        if (abs(a-b)>hb):
            return 0.5*(a-b)**2
        else:
            return 0.5*(abs(a-b)-0.5*1)
    
    def tensor_huber_lose(self, a,b):
        #TODO
        res = torch.FloatTensor([]).to(self.device)
        ca = a.clone().view(-1)
        cb = b.clone().view(-1)
        for i in range(0, len(a)):
            torch.concat([res,self.huber_lose(ca[i],cb[i])])
        return res
    
    def cal_rule_loss(self, state, mu, advantage):
        # 用于训练网络直接学习比例引导法，直接给出loss
        comp = torch.zeros((len(advantage),1),device=self.device).detach()
        #lose = F.mse_loss(-torch.min(advantage,comp)*mu,-torch.min(advantage,comp)*self.get_rule(state))
        lose = torch.mean(-torch.min(advantage,comp)*F.huber_loss(mu,self.get_rule(state), reduction='none'))
        #self.tensor_huber_lose(mu,self.get_rule(state))
        return lose

    def policy(self, state):
        """
        给出具体动作
        """
        state = torch.FloatTensor(np.array([state])).to(self.device)
        if state[0][0]<=1 and random.random()<1:
            #此处根据比例导引法给出动作，用于验证环境的合法性，训练过程中禁用此部分。
            s = state[0].tolist()
            v = s[2]*1000
            seta = s[5]*math.pi
            qe_dot = s[6]
            qb_dot = s[7]
            Ny = qe_dot*v/(9.8*40)#float(np.clip(qe_dot*v/(9.8*40),-1.0,1.0))
            Nz = qb_dot*v*math.cos(seta)/(9.8*40)#float(np.clip(qb_dot*v*math.cos(seta)/(9.8*40),-1.0,1.0))
            return [Ny*3,Nz*3]
        else:
            mu, std = self.actor(state)
            #writer.add_graph(self.actor,input_to_model=state,verbose=True)
            action_dist = torch.distributions.Normal(mu, std)
            action = torch.clamp(action_dist.sample(),-1,1)
            #action = action_dist.sample()
            return [action[0][0].item(), action[0][1].item()]
    

    def update(self, samples: deque, train_cnt:int):
        """
        训练过程
        """
        self.count += 1
        state, action, reward, next_state, done = zip(*samples)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.tensor(action).view(-1, 2).to(self.device)
        reward = torch.tensor(reward).view(-1, 1).to(self.device)
        #reward = (reward - reward.mean())/(reward.std() + 1e-10)  # 和TRPO一样,对奖励进行修改,方便训练
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).view(-1, 1).to(self.device)
        
        ######################
        #计算累计奖励和进步值
        td_target = compute_advantage(self.gamma,self.lmbda, reward.cpu(), done.cpu()).to(self.device)
        advantage = td_target-self.critic(state).detach()
        ######################

        writer.add_scalar("平均advantage", advantage.mean(), train_cnt)
        writer.add_scalar("最大advantage", advantage.max(), train_cnt)
        writer.add_scalar("最小advantage", advantage.min(), train_cnt)

        mu, std = self.actor(state)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())

        writer.add_scalar('平均偏差', std.mean(), train_cnt)
        old_log_probs = action_dists.log_prob(action)
        old_log_probs_dot = (old_log_probs[:,0]*old_log_probs[:,1]).view(-1,1)

        for ppoUpdateCnt in range(self.ppo_epochs):
            mu, std = self.actor(state)

            #PPO#######################
            action_dists = torch.distributions.Normal(mu, std)
            log_prob = action_dists.log_prob(action)
            log_probs_dot = (log_prob[:,0]*log_prob[:,1]).view(-1,1)
            # e(log(a/b))
            ratio = torch.exp(log_prob - old_log_probs)
            #ratio = torch.exp(log_probs_dot - old_log_probs_dot)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            PPOclip  = torch.mean(torch.min(surr1, surr2))
            ###########################
            
            #ent######################
            ent = torch.log(torch.sqrt(2*3.1415926*math.exp(1)*std))
            ##########################

            #GSIL######################
            comp = torch.zeros((len(advantage),1),device=self.device).detach()
            gsil = torch.mean(torch.max(advantage,comp)*F.huber_loss(mu,torch.clamp(action,-1,1),reduction="none",delta=1))
            #gsil = F.mse_loss(torch.max(advantage,comp)*mu,torch.max(advantage,comp)*torch.clamp(action,-1,1).detach())
            ###########################
            
            #直接学习比例引导法######
            rule_loss = self.cal_rule_loss(state, mu, advantage)
            ############################

            #损失回传
            actor_loss = torch.mean(-PPOclip-self.weight2*ent)+self.weight3*gsil+self.weight4*rule_loss
            critic_loss = torch.mean(
                F.huber_loss(self.critic(state), td_target.detach(), delta=1.0)
            )

            if ppoUpdateCnt == 0:
                writer.add_scalar("PPO", torch.mean(PPOclip), train_cnt)
                writer.add_scalar("ent", torch.mean(ent), train_cnt)
                writer.add_scalar("gsil", gsil, train_cnt)
                writer.add_scalar("ruleloss", rule_loss, train_cnt)
                writer.add_scalar("actor_loss", actor_loss, train_cnt)
                writer.add_scalar("critic_loss", critic_loss, train_cnt)
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()

        return actor_loss





class replayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append( (state, action, reward, next_state, done) )
    
    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

class actionBuffer:
    def __init__(self, capacity: int):
        if (capacity == 0):
            self.capacity = 0
        else:
            self.capacity = capacity
            self.buffer = deque(maxlen = capacity)
            for i in range(0, capacity):
                self.buffer.append([0,0])
    
    def delay(self, newaction):
        if (self.capacity == 0):
            return newaction
        oldaction = self.buffer.pop()
        self.buffer.append(newaction)
        return oldaction     
    
def play(env, env_agent, cfg, episode_count=1):
    """
    此函数用于检测模型的表现，检测100次仿真中的命中率，不对模型进行更新
    """
    acc = 0
    trueacc = 0
    rewards = []
    episode_reward = 0
    for e in range(episode_count):
        s, _ = env.reset()
        done = False
        episode_cnt = 0
        while not done:
            env.render()
            try:
                a = env_agent.policy(s)
            except:
                return 0
            n_state, reward, done, _, truehit, _ = env.step(a)
            episode_reward += reward
            episode_cnt += 1
            s = n_state
            if (done):
                if (n_state[0]*5000<=10):
                    acc += 1
                if truehit:
                    trueacc += 1
                rewards.append(episode_reward)
                break
    
    print(f'Get reward {math.fsum(rewards)/episode_count}. Acc {acc} times, TureAcc {trueacc} times')
    env.close()
    return acc


class Config:
    """
    超参，部分超参有PBTgene决定 
    """
    num_episode = 10000                  #训练次数
    state_dim = 16                      #输入向量大小
    hidden_layers_dim = [ 128, 128 ]    #网络隐藏层结构
    action_dim = 2                      #动作维度
    actor_lr = 1e-4                     #actor学习率
    critic_lr = 5e-3                    #critic学习率   
    PPO_kwargs = {
        'lmbda': 1,                     
        'eps': 0.1,                     #PPO截断
        'ppo_epochs': 10                #样本重复学习次数
    }
    gamma = 0.9                        #奖励衰减因子
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    buffer_size = 40000
    batch_size = 40000
    save_path = r'轨迹/2/ac_model.ckpt'
    # 回合停止控制
    max_episode_rewards = 499.5
    max_episode_steps = 10000
    
    max_norm_value = 8
    
    test_episode = 1                    #样本测试次数
    weight2 = 0.0001
    weight3 = 1
    
    gene_index = 0
    
    def __init__(self, env):
        # self.state_dim = env.observation_space.shape[0]
        # try:
        #     self.action_dim = env.action_space.n
        # except Exception as e:
        #     self.action_dim = env.action_space.shape[0]
        print(f'device={self.device} | env={str(env)}')
        
    def read_gene(self, gene, index):
        """
        从PBT基因中读取超参
        """ 
        
        self.actor_lr = gene['actor_lr']
        self.critic_lr = gene['critic_lr']
        self.gamma = gene['gamma']
        self.buffer_size = gene['buffer_size']
        self.max_episode_steps = gene['max_episode_steps']
        self.weight2 = gene['weight2']
        self.weight3 = gene['weight3']
        self.train_cnt = gene["train_cnt"]
        self.stage = gene['stage']
        self.last_cnt = gene['last_cnt']

        self.num_episode = PBT_gene.check_point
        self.gene_index = index
        self.save_path = r"traj/2/ac{}_model.ckpt".format(index)
        self.save_path_cc = r"traj/2/cc{}_model.ckpt".format(index)
        return self
        
class PBT_gene():
    init_gene = {                   #模板基因
        "actor_lr": 1e-4,
        "critic_lr": 5e-3,
        "gamma": 0.99,
        "buffer_size": 10000,
        "max_episode_steps": 1000,
        "weight2": 0.0001,
        "weight3": 1,
    }
    
    
    mutation_rate = 0#0.2             #变异概率
    mutation_gap = 0.2              #变异幅度
    replace_number = 0              #淘汰个数
    capacity = 1                    #种群大小
    check_point = 1000             #每隔多少次训练迭代一次
    
    total_PBT_max_steps = 10000     #总训练次数
    
    genes = []
    score = []
    
    def __init__(self) -> None:
        self.creat_pack()

        try:
            for i in range(self.capacity):
                with open('轨迹/2/gene{}.txt'.format(i), "r") as f:
                    for keys in self.init_gene.keys():
                        self.genes[i][keys] = float(f.readline())
                        if keys in ['buffer_size', 'max_episode_steps']:
                            self.genes[i][keys] = int(self.genes[i][keys])
                    self.genes[i]['train_cnt'] = int(f.readline())
                    self.genes[i]['stage'] = int(f.readline())
                    self.genes[i]['last_cnt'] = int(f.readline())
        except:
            pass
    
    def creat_pack(self):
        for i in range(self.capacity):
            self.genes.append(self.mutate(self.init_gene.copy()))
            self.genes[i]['train_cnt'] = 0
            self.genes[i]['stage'] = 0
            self.genes[i]['last_cnt'] = 0
            self.score.append(0)
    
    def mutate(self, gene):
        """
        基因变异
        """
        for key in gene:
            if random.random()<self.mutation_rate:
                gene[key] = np.random.normal(gene[key],self.mutation_gap*gene[key])
                if (key in ["max_episode_steps",'buffer_size']):
                    gene[key] = math.floor(gene[key])
                if (key in ["gamma"]):
                    gene[key] = min(gene[key],0.99)
        return gene
    
    def evolution(self):
        sorted_score = sorted(self.score, reverse=True)
        sorted_index = []
        for num in sorted_score:
            for j, i in enumerate(self.score):
                if i == num and not j in sorted_index:
                    sorted_index.append(j)
                    break

        #self.genes = [self.genes[i] for i in sorted_index]
        for i in range(self.replace_number):
            if (sorted_score[sorted_index[-i-1]]<np.mean(score)):
                target_index = sorted_index[random.randint(0,self.replace_number-1)]
                self.genes[sorted_index[-i-1]] = self.mutate(self.genes[target_index])
                shutil.copy(r"traj/2/ac{}_model.ckpt".format(target_index), r"traj/2/ac{}_model.ckpt".format(sorted_index[-i-1]))
                shutil.copy(r"traj/2/cc{}_model.ckpt".format(target_index), r"traj/2/cc{}_model.ckpt".format(sorted_index[-i-1]))
    
    def get_score(self, index, score):
        self.score[index] = score

    def write_gene_to_file(self, file, index):
        with open(file, "w") as f:
            for keys in self.init_gene.keys():
                f.write(str(self.genes[index][keys])+"\n")
            f.write(str(self.genes[index]['train_cnt'])+"\n")
            f.write(str(self.genes[index]['stage'])+"\n")
            f.write(str(self.genes[index]['last_cnt'])+"\n")

def drawTraj(m, t, index):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(index):
        ax.scatter(m['x'][i], m['z'][i],m['y'][i],s=1,color='r')
        ax.scatter(t['x'][i], t['z'][i],t['y'][i],s=1,color='b')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    plt.show()

def train_agent(env, cfg):
    """
     训练主函数
    """
    curr = curriclum()
    curr.stage = cfg.stage
    curr.last_cnt = cfg.last_cnt
    stage_inf = curr.get_stage()
    
    ac_agent = PPO(
        state_dim=cfg.state_dim,
        hidden_layers_dim=cfg.hidden_layers_dim,
        action_dim=cfg.action_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        PPO_kwargs=cfg.PPO_kwargs,
        device=cfg.device,
        max_norm = cfg.max_norm_value,
        weight2 = cfg.weight2,
        weight3= cfg.weight3,
        weight4 = stage_inf['weight4'],
    )         
    try:
        #尝试读取历史模型
        ac_agent.actor.load_state_dict(torch.load(cfg.save_path))
        ac_agent.critic.load_state_dict(torch.load(cfg.save_path_cc))
    except: 
        pass
    tq_bar = tqdm(range(cfg.num_episode))
    rewards_list = [0 for i in range(9)]
    now_reward = 0
    bf_reward = -np.inf
    buffer_ = replayBuffer(cfg.buffer_size)
    train_cnt = cfg.train_cnt
    tran_acc = 0 #命中计数，用于计算命中率
    true_hit = 0
    done_cnt = 0 #完成计数， 用于计算命中率
    trajM = {'x':[],'y':[],'z':[]}
    trajT = {'x':[],'y':[],'z':[]}
    trajIndx = -1
    for i in tq_bar:
        #如果网络中出现nan
        if (torch.isnan(ac_agent.actor.fc_mu.weight).sum()>0):
            print("nan erro")
        tq_bar.set_description(f'Episode pars1 [ {i+1} / {cfg.num_episode} ]| gene {cfg.gene_index}') 
        iterReward = 0
        interMinD = 0
        s, d = env.reset(stage_inf = stage_inf)
        done = False
        episode_rewards = 0
        steps = 0
        minD = d*5000
        actionbuffer = actionBuffer(stage_inf['delay']) #延迟操作
        for key in ['x','y','z']:
            trajM[key].append([])
            trajT[key].append([])
        trajIndx += 1
        while not done:
            a = ac_agent.policy(s)
            ra = actionbuffer.delay(a)
            n_s, r, done, d, truehit, _ = env.step(ra)
            state = env.state
            for index, key in enumerate(['x','y','z']):
                trajM[key][trajIndx].append(state[index+1])
                trajT[key][trajIndx].append(state[index+7])
            buffer_.add(s, a, r, n_s, done)
            s = n_s
            minD = min(d*5000,minD)
            episode_rewards += r
            steps += 1
            if (steps >= cfg.max_episode_steps):
                break
        iterReward += episode_rewards
        interMinD += minD
        if (minD<10):
            tran_acc = tran_acc+1.0
        if truehit:
            true_hit += 1
        done_cnt = done_cnt+1.0
        if (len(buffer_)>=cfg.buffer_size): #仅当buffer被填满后才进行训练
            train_cnt = train_cnt+1
            #actor_loss = ac_agent.update(buffer_.buffer, train_cnt)
            actor_loss = 0
            buffer_ = replayBuffer(cfg.buffer_size) #buffer清空
            writer.add_scalar("命中率",tran_acc/done_cnt,train_cnt)
            writer.add_scalar("逆轨命中率",true_hit/done_cnt,train_cnt)
            writer.add_scalar("最小距离",interMinD, train_cnt )
            writer.add_scalar("奖励", now_reward, train_cnt)
            writer.add_scalar("环境难度", curr.stage, train_cnt)
            if (curr.is_meet(tran_acc/done_cnt,actor_loss,train_cnt)):
                curr.nextstage(train_cnt)
                ac_agent.weight4 = curr.get_stage()['weight4']
                stage_inf = curr.get_stage()
                
            tran_acc = 0
            done_cnt = 0
            true_hit = 0
        rewards_list.append(iterReward)
        now_reward = np.mean(rewards_list[-10:])
        if bf_reward <= now_reward: #若最近几次奖励由于历史记录，保存当前模型
            torch.save(ac_agent.actor.state_dict(), cfg.save_path)
            torch.save(ac_agent.critic.state_dict(), cfg.save_path_cc)
            bf_reward = now_reward
        
        tq_bar.set_postfix({'minD':f'{interMinD:.2f}','lastMeanRewards': f'{now_reward:.2f}', 'BEST': f'{bf_reward:.2f}'})
    env.close()
    #drawTraj(trajM, trajT, trajIndx)
    return train_cnt, ac_agent, curr






if __name__ == '__main__':
    print('=='*35)
    print('3D M-T')
    env = OneVOneEnv()#gym.make('Pendulum-v1')
    
    #####PBT流程
    PBTgene = PBT_gene() #生成PBT基因库
    writer = SummaryWriter(r"traj/2/logs")
    train_cnt = 0
    #while 1:
    #for evol_time in range(0,int(PBTgene.total_PBT_max_steps/PBTgene.check_point)):
    while train_cnt < PBTgene.total_PBT_max_steps:
        for index in range(PBTgene.capacity):
            PBTgene.write_gene_to_file(r"traj/2/gene{}.txt".format(index), index)   #将基因配置保持在文件里
            cfg = Config(env).read_gene(PBTgene.genes[index], index)
            train_cnt, ac_agent, curr = train_agent(env, cfg)                     #训练过程，ac_agent是agent类，train_cnt保存已经完成的训练次数
            PBT_gene.genes[index]['train_cnt'] = train_cnt
            PBT_gene.genes[index]['stage'] = curr.stage
            PBT_gene.genes[index]['last_cnt'] = curr.last_cnt
            torch.save(ac_agent.actor.state_dict(), r"traj/2/tmp.ckpt")
            torch.save(ac_agent.critic.state_dict(), r"traj/2/tmpcc.ckpt")
            score_fin = play(env, ac_agent, cfg)
            ac_agent.actor.load_state_dict(torch.load(cfg.save_path))
            score_best = play(env, ac_agent, cfg)
            #if (score_fin>=score_best):
            if (1):                                                         #仅保存每个训练过程最后的模型而不是最优的模型
                ac_agent.actor.load_state_dict(torch.load(r"traj/2/tmp.ckpt"))
                ac_agent.critic.load_state_dict(torch.load(r"traj/2/tmpcc.ckpt"))
                torch.save(ac_agent.actor.state_dict(), cfg.save_path)
                torch.save(ac_agent.critic.state_dict(), cfg.save_path_cc)
            score = max(score_best,score_fin)
            PBTgene.get_score(index, score)
        
        PBTgene.evolution()                                                  #PBT优胜劣汰
    
    writer.close()
