# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pyplot as pl
from copy import deepcopy as dc
import numpy as np
import random
import pandas as pd
import statsmodels.formula.api as smf
import networkx as nx
import time
pl.rcParams.update({'font.size': 12, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})

class Diffuse: #默认网络结构为节点数量为10000，边为60000的单向随机网络
    repetes = 10 #默认多次扩散重复次数
    def __init__(self, p, q, num_runs, G=nx.gnm_random_graph(10000, 30000)):
        if not nx.is_directed(G):
            self.G = G.to_directed()
        else:
            self.G = G
        self.p = p
        self.q = q
        self.num_runs = num_runs
        
    def decision(self, i): #个体状态决策规则
        influ = sum([1 for k in self.G.predecessors(i) if self.G[k]['state']])           
        prob = self.p + self.q * influ
        return True if random.random() <= prob else False
        
    def single_diffuse(self): #单次扩散
        for i in self.G.nodes():
            self.G[i]['state'] = False

        non_adopt_set = [i for i in self.G]
        num_of_adopt = []

        for j in range(self.num_runs):                                 
            x = 0
            random.shuffle(non_adopt_set)
            for i in non_adopt_set:                   
                if self.decision(i):
                    self.G[i]['state'] = 1
                    non_adopt_set.remove(i)
                    x += 1
            num_of_adopt.append(x)

        return num_of_adopt
    
    def repete_diffuse(self): #多次扩散
        adopt_cont = []
        for i in range(self.repetes):
            num_of_adopt = self.single_diffuse()
            adopt_cont.append(num_of_adopt)
        return adopt_cont
        
class Random_Grid_Search:
    t_n = 500  # 抽样量
    c_n = 50  # 保留参数量
    threshold = 1e-6  # 循环停止阈值
    orig_points = []  # 初始化边界点
    def __init__(self,s):  # 初始化实例参数
        self.s = np.array(s)  # 待拟合曲线
        self.s_len = len(s)
        self.para_range = [[1e-6, 0.1], [1e-3, 1],[sum(s), 4 * sum(s)]]  # 参数范围
        self.p_range = [[1e-6, 0.1],[1e-3, 1],[sum(s), 4 * sum(s)]]  # 用于产生边界节点的参数范围 
    
    def gener_orig(self): #递归产生边界点
        if len(self.p_range) == 0:
            return
        else:  
            pa = self.p_range[-1]
            if self.orig_points == []:
                self.orig_points = [[pa[0]], [pa[1]]]  # 初始化,排除orig_points为空的情形
            else:
                self.orig_points = [[pa[0]] + x for x in self.orig_points] + [[pa[1]]+x for x in self.orig_points]  # 二分裂
            self.p_range.pop()
            return self.gener_orig()
    
    def sample(self, c_range):  # 抽样参数点
        p_list = []
        for pa in c_range:
            if isinstance(pa[0],float):
                x = (pa[1]-pa[0])*np.random.random(self.t_n) + pa[0]
            else:
                x = np.random.randint(low=pa[0],high=pa[1]+1,size=self.t_n)
            p_list.append(x)

        p_list = np.array(p_list).T
        return p_list.tolist()
    
    def f(self, params): #扩散函数
        diffu_cont = np.zeros(self.s_len)
        t_list = np.arange(1,self.s_len+1)
        a = np.array([1 - np.exp(-(params[0] + params[1]) * t) for t in t_list])
        b = np.array([1 + params[1] / params[0] * np.exp(-(params[0] + params[1]) * t) for t in t_list])
        diffu_cont = params[2] * a / b

        adopt_cont = np.zeros_like(diffu_cont)
        adopt_cont[0] = diffu_cont[0]
        for t in range(1,self.s_len):
            adopt_cont[t] = diffu_cont[t] - diffu_cont[t-1]

        return adopt_cont
    
    def r2(self, params):
        f_act = self.f(params)
        tse = np.sum(np.square(self.s - f_act))
        mean_y = np.mean(self.s)
        ssl = np.sum(np.square(self.s - mean_y))
        R_2 = (ssl - tse) / ssl
        return R_2

    def mse(self,params):  #定义适应度函数（mse）
        a = self.f(params)
        sse = np.sum(np.square(self.s - a))
        return np.sqrt(sse) / self.s_len  # 均方误
    
    def optima_search(self):
        self.gener_orig()
        c_range = dc(self.para_range)
        samp = self.sample(c_range)
        solution = sorted([self.mse(x)] + x for x in samp + self.orig_points)[:self.c_n]
        for u in range(100):
            params_min = np.min(solution,0)[1:]  # 最小值
            params_max = np.max(solution,0)[1:]  # 最大值
            c_range = [[params_min[j], params_max[j]] for j in range(params_min.size)]  # 重新定界
            samp = self.sample(c_range)
            solution = sorted([[self.mse(x)] + x for x in samp] + solution)[:self.c_n]
            r = sorted([x[0] for x in solution])
            v = (r[-1] - r[0]) / r[0]
            if v < self.threshold:        
                break
        else:
            print 'Searching ends in 100 runs'
            break

        return solution[0]  #sse,p,q,m
        
class Estimate_ABM:
    num_conds = 2  # 构建网格的节点个数
    k = 6
    def __init__(self, s, intv_p=0.001, intv_q = 0.005, G=nx.gnm_random_graph(10000, 30000)):
        self.s = s
        self.s_len = len(s)
        self.intv_p = 0.001
        self.intv_q = 0.005
        self.G = G
    
    def r2(self,f_act):
        f_act = np.array(f_act)
        tse = np.sum(np.square(self.s - f_act))
        mean_y = np.mean(self.s)
        ssl = np.sum(np.square(self.s - mean_y))
        R_2 = (ssl - tse) / ssl
        return R_2
    
    def get_M(self,p,q):  # 获取对应扩散率曲线的最优潜在市场容量
        diffu = Diffuse(p, q, self.s_len, G=self.G)
        s_estim =diffu.repete_diffuse()
        x = np.mean(s_estim, axis=0)
        a = np.sum(np.square(x)) / np.sum(self.s)  # 除以np.sum(self.s)是为减少a的大小
        b = -2 * np.sum(x * self.s) / np.sum(self.s)
        c = np.sum(np.square(self.s)) / np.sum(self.s)
        mse, sigma = np.sqrt(sum(self.s) * (4 * a * c - b ** 2) / (4 * a * self.s_len)), -b / (2 * a)
        m = sigma * self.G.number_of_nodes()
        return mse,p,q,m,x*sigma
    
    def gener_p0_q0(self):  # 生成初始搜索点(p0,q0)
        rgs = Random_Grid_Search(self.s)
        P0,Q0 = rgs.optima_search()[1:3]  # SABM最优点（P0,Q0）
        p_range = np.linspace(0.4 * P0, P0, num=3)
        q_range = np.linspace(0.2 * Q0 / self.k, 0.6 * Q0 / self.k, num=3)
        to_fit = {}    
        params_cont = []
        for p in p_range:
            for q in q_range:
                diffu = Diffuse(p, q, self. s_len, self.G)
                s_estim = diffu.repete_diffuse()
                s_estim_avr = np.mean(s_estim,axis=0)
                rgs_1 = Random_Grid_Search(s_estim_avr)
                P, Q = rgs_1.optima_search()[1:3]
                params_cont.append([[p, q], [P, Q]])
        
        to_fit['p'] = [x[0][0] for x in params_cont]
        to_fit['q'] = [x[0][1] for x in params_cont]
        to_fit['P'] = [x[1][0] for x in params_cont]
        to_fit['Q'] = [x[1][1] for x in params_cont]
        to_fit = pd.DataFrame(to_fit)

        result_p = smf.ols('p~P+Q-1', data=to_fit).fit()
        result_q = smf.ols('q~P+Q-1', data=to_fit).fit()

        p0 = result_p.params['P'] * P0 + result_p.params['Q'] * Q0
        q0 = result_q.params['P'] * P0 + result_q.params['Q'] * Q0
        return p0, q0
    
    def solution_search(self, p0, q0):
        solution_cont = []
        diff_cont = []
        pq_cont = []
        
        for p in (p0 - self.intv_p, p0, p0 + self.intv_p):
            for q in (q0 - self.intv_q, q0, q0 + self.intv_q):
                p,q = round(p, 4), round(q, 4)  # 防止出错
                pq_cont.append((p, q))
                solution = self.get_M(p, q)
                solution_cont.append(solution[:4]) # M_sse,p,q,s_M
                diff_cont.append(solution[4])  # x

        best_solution = sorted(solution_cont)[:self.num_conds] 
        while 1:
            solution_cont2 = []
            diff_cont2 = []
            pq_cont2 = []
            
            for z in best_solution:
                temp = [(z[1]-self.intv_p,z[2]-self.intv_q),(z[1],z[2]-self.intv_q),(z[1]+self.intv_p,z[2]-self.intv_q),
                                (z[1]-self.intv_p,z[2]),       (z[1],z[2]),       (z[1]+self.intv_p,z[2]),
                                (z[1]-self.intv_p,z[2]+self.intv_q),(z[1],z[2]+self.intv_q),(z[1]+self.intv_p,z[2]+self.intv_q)]
                pq_cont2.extend(temp)
            pq_cont2 = list(set(pq_cont2+pq_cont))
            
            for y in pq_cont2:
                if y in pq_cont:
                    solution_cont2.append(solution_cont[pq_cont.index(y)])
                    diff_cont2.append(diff_cont[pq_cont.index(y)])
                else:
                    solution = self.get_M(y[0],y[1])
                    solution_cont2.append(solution[:4])
                    diff_cont2.append(solution[4])

            best_solution = sorted(solution_cont2)[:self.num_conds]
            opt_solution = best_solution[0]
            opt_curve = diff_cont2[solution_cont2.index(opt_solution)]

            if len(pq_cont2) == len(pq_cont):
                break
            else:
                solution_cont = solution_cont2
                diff_cont = diff_cont2
                pq_cont = pq_cont2

        f_act = opt_curve
        R2 =self.r2(f_act)
        search_steps = len(pq_cont)
        result = {'params': opt_solution[1:], 'fitness': R2, 'best_curve': f_act, 'steps': search_steps,'path': pq_cont,}  # [p,q,m],拟合曲线,搜索步数,搜索范围
        return result #[p,q,m],拟合曲线,搜索步数,搜索范围
        
if __name__ == '__main__':
    data_set = {'room air conditioners':(np.arange(1949,1962),[96,195,238,380,1045,1230,1267,1828,1586,1673,1800,1580,1500]),
            'color televisions':(np.arange(1963,1971),[747,1480,2646,5118,5777,5982,5962,4631]),
            'clothers dryers':(np.arange(1949,1962),[106,319,492,635,737,890,1397,1523,1294,1240,1425,1260,1236]),
            'ultrasound':(np.arange(1965,1979),[5,3,2,5,7,12,6,16,16,28,28,21,13,6]),
            'mammography':(np.arange(1965,1979),[2,2,2,3,4,9,7,16,23,24,15,6,5,1]),
            'foreign language':(np.arange(1952,1964),[1.25,0.77,0.86,0.48,1.34,3.56,3.36,6.24,5.95,6.24,4.89,0.25]),
            'accelerated program':(np.arange(1952,1964),[0.67,0.48,2.11,0.29,2.59,2.21,16.80,11.04,14.40,6.43,6.15,1.15])}

    china_set = {'color televisions':(np.arange(1997,2013),[2.6,1.2,2.11,3.79,3.6,7.33,7.18,5.29,8.42,5.68,6.57,5.49,6.48,5.42,10.72,5.15]),
             'mobile phones':(np.arange(1997,2013),[1.7,1.6,3.84,12.36,14.5,28.89,27.18,21.33,25.6,15.88,12.3,6.84,9.02,7.82,16.39,7.39]),
             'computers':(np.arange(1997,2013),[2.6,1.2,2.11,3.79,3.6,7.33,7.18,5.29,8.42,5.68,6.57,5.49,6.48,5.42,10.72,5.15]),
             'conditioners':(np.arange(1992,2013),[1.19,1.14,2.67,3.09,3.52,4.68,3.71,4.48,6.32,5.0,15.3,10.69,8.01,10.87,7.12,7.29,5.2,6.56,5.23,9.93,4.81]),
             'water heaters':(np.arange(1988,2013),[28.07,8.4,5.86,6.37,3.9,4.08,5.42,4.12,3.45,3.31,3.12,1.64,2.36,1.8,5.48,1.35,1.47,0.52,1.03,3.28,-1.4,1.72,1.26,0.62,1.25])
             }
             
    est_dict = {}
    for k in sorted(data_set.keys()):
        print '=====================================%s====================================='%k
        time1 = time.clock()
        s = data_set[k][1]
        est_abm = Estimate_ABM(s)
        p0,q0 = est_abm.gener_p0_q0()
        estims,f_act,R2,steps,pq_cont = est_abm.solution_search(p0,q0)
        est_dict[k] = {'p':estims[0],'q':estims[1],'m':estims[2],'curve':f_act,'r2':R2,'path':pq_cont}
        print '    Time elasped: %.2f s'%(time.clock()-time1),'s'
        print '    R2:%.4f    steps:%s'%(R2,steps)
