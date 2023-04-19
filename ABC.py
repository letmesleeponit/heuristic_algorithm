import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Callable
'''
在標準的ABC演算法中，
雇用蜂利用先前的蜜源資訊尋找新的蜜源並與觀察蜂分享蜜源資訊；
觀察蜂在蜂房中等待並依據採蜜蜂分享的資訊尋找新的蜜源；
偵查蜂的任務是尋找一個新的有價值的蜜源，它們在蜂房附近隨機地尋找蜜源。
'''

### objective function: test1(Rastrigin function)
def Rastrigin_function(val:List[int], dim:int) -> int:
    obj = 0
    for dim_ite in range(dim): 
        obj += (val[:, dim_ite] ** 2 - 10 * np.cos(2 * np.pi * val[:, dim_ite]))
    obj += dim * 10
    return obj

### objective function: test2(STYBLINSKI-TANG function)
def Styblinski_Tang_fun(val:List[int], dim:int) -> int:
    obj = 0
    for dim_ite in range(dim):
        obj += val[:, dim_ite] ** 4 - 16 * val[:, dim_ite] ** 2 + 5 * val[:, dim_ite]
    obj /= 2
    return obj

class ABC:
    '''
    Parameter:
    iteration : 迭代次數 
    fun : 目標式
    dimension : 問題維度
    num : 蜜蜂總數
    limit : 偵查蜜源上限，用以決定是否放棄蜜源
    range_min : 解的下限
    range_max : 解的上限
    obj_sta : min or max
    pos : 蜜蜂的位置
    trial : 偵查蜜源的次數
    '''
    def __init__(self, iteration:int, fun:Callable[[List[int], int], int], dimension:int, num:int, limit:int, range_min:List[int], range_max:List[int], obj_sta:str):
        # parameter record
        self.iteration = iteration
        self.fun = fun
        self.dimension = dimension
        self.num = num
        self.limit = limit
        self.range_min = range_min
        self.range_max = range_max
        self.obj_sta = obj_sta
        self.pos = np.add(range_min, np.subtract(range_max, range_min) * np.random.random(size=(num, dimension)))
        self.trial = np.zeros(num)

        # result record
        self.rec_val = []
        self.rec_pos = []

    # 將目標函數傳回的值丟入ABC內設的蜜源豐富程度公式
    def get_fit_val(self, fun_val:np.array, sta:bool):
        rec_arr = []
        if self.obj_sta == 'min':
            for val in fun_val:
                rec_arr.append(1 / (1 + val) if val >= 0 else 1 + abs(val))
        else:
            print('Under construction')
            print('Program break')
            exit()

        if sta:
            return rec_arr[-1]
        else:
            self.fit_val = rec_arr


    # 雇用蜂階段 => 每個雇用蜂皆有一個固定守著的位置，並在附近環繞，看看有沒有表現更好的解
    def employed_bee(self) -> None:
        for sol_index in range(self.num):
            # 找另一個蜜源的位置(不可為此處for迴圈迭代的蜜源位置)
            sol_compare_loc = np.random.randint(0, self.num)
            while sol_compare_loc == sol_index:
                sol_compare_loc = np.random.randint(0, self.num)
            
            # 根據這個選出來的蜜源位置，於此處蜜源附近找另一點
            pos_new = self.pos[sol_index].copy()
            change_loc = np.random.randint(self.dimension)
            pos_new[change_loc] = pos_new[change_loc] + np.random.uniform(-1,1) * (pos_new[change_loc] - self.pos[sol_compare_loc][change_loc])
            pos_new = np.clip(pos_new, self.range_min, self.range_max) # 讓數值介於一範圍

            # 比較這兩點蜜源誰的蜜比較多(fit value最高)，並更新為最高的
            ori_obj = self.fun(np.expand_dims(self.pos[sol_index], axis=0), self.dimension)
            new_obj = self.fun(np.expand_dims(pos_new, axis=0), self.dimension)
            if self.obj_sta == 'min':
                if new_obj < ori_obj:
                    self.pos[sol_index] = pos_new
                else:
                    self.trial[sol_index] += 1
            else:
                if new_obj > ori_obj:
                    self.pos[sol_index] = pos_new
                else:
                    self.trial[sol_index] += 1

    # 觀察蜂階段 => 與雇用蜂動作類似，皆是尋找蜜源附近有沒有表現更好的解，但觀察蜂會更傾向於搜尋雇用蜂發現的表現較好的解
    def onlooker_bee(self) -> None:
        for sol_index in range(self.num):
            val = self.fun(np.expand_dims(self.pos[sol_index], axis=0), self.dimension)
            if np.random.uniform() < self.get_fit_val(val, True):
                # 找另一個蜜源的位置(不可為此處for迴圈迭代的蜜源位置)
                sol_compare_loc = np.random.randint(0, self.num)
                while sol_compare_loc == sol_index:
                    sol_compare_loc = np.random.randint(0, self.num)
                
                # 根據這個選出來的蜜源位置，於此處蜜源附近找另一點
                pos_new = self.pos[sol_index].copy()
                change_loc = np.random.randint(self.dimension)
                pos_new[change_loc] = pos_new[change_loc] + np.random.uniform(-1,1) * (pos_new[change_loc] - self.pos[sol_compare_loc][change_loc])
                pos_new = np.clip(pos_new, self.range_min, self.range_max) # 讓數值介於一範圍

                # 比較這兩點蜜源誰的蜜比較多(fit value最高)，並更新為最高的
                ori_obj = self.fun(np.expand_dims(self.pos[sol_index], axis=0), self.dimension)
                new_obj = self.fun(np.expand_dims(pos_new, axis=0), self.dimension)
                if self.obj_sta == 'min':
                    if new_obj < ori_obj:
                        self.pos[sol_index] = pos_new
                    else:
                        self.trial[sol_index] += 1
                else:
                    if new_obj > ori_obj:
                        self.pos[sol_index] = pos_new
                    else:
                        self.trial[sol_index] += 1

    # 偵查蜂階段 => 主要工作在於判斷每個蜜源是否達到採集(測試)上限，如果達到便會放棄該蜜源，另外生成一蜜源代替 
    def scout_bee(self) -> None:
        for sol_index in range(self.num):
            if self.trial[sol_index] > self.limit:
                self.pos[sol_index] = self.range_min + np.subtract(self.range_max, self.range_min) * np.random.random(size=(1, dimension))
                self.trial[sol_index] = 0

    # 評估目前所得結果為何
    def evaluation(self) -> None:
        try:
            self.fun_val = self.fun(self.pos, self.dimension)
            if self.obj_sta == 'min':
                if min(self.fun_val) > self.rec_val[-1]:
                    self.rec_val.append(self.rec_val[-1])
                    self.rec_pos.append(self.rec_pos[-1])
                else:
                    self.rec_val.append(min(self.fun_val))
                    self.rec_pos.append(self.pos[np.argmin(self.fun_val)])
            elif self.obj_sta == 'max':
                if max(self.fun_val) < self.rec_val[-1]:
                    self.rec_val.append(self.rec_val[-1])
                    self.rec_pos.append(self.rec_pos[-1])
                else:
                    self.rec_val.append(max(self.fun_val))
                    self.rec_pos.append(self.pos[np.argmax(self.fun_val)])
        except:
            self.rec_val.append(min(self.fun_val)) if self.obj_sta == 'min' else self.rec_val.append(max(self.fun_val))
            self.rec_pos.append(self.pos[np.argmin(self.fun_val)]) if self.obj_sta == 'min' else self.rec_pos.append(self.pos[np.argmax(self.fun_val)])
            
    # pipeline
    def ite_run(self) -> None:
        for _ in range(self.iteration):
            self.onlooker_bee()
            self.scout_bee()
            self.evaluation()


iteration = 100 # 迭代次數
dimension = 2 # 問題維度
range_min = [-5] * dimension # 解的下限
range_max = [5] * dimension # 解的上限
num = 200 # 蜜蜂總數
limit = 10 # 偵查蜜源上限，用以決定是否放棄蜜源
obj_sta = 'min' # min or max

sta_time = time.time()
obj = ABC(iteration, Rastrigin_function, dimension, num, limit, range_min, range_max, obj_sta)
obj.ite_run()
end_time = time.time()

print(obj.rec_pos[-1])
print(obj.rec_val[-1])
print(f'Total consumse time is {end_time - sta_time: .2f}')
plt.plot(obj.rec_val)
plt.title('The convergence histories of ABC')
plt.xlabel('The iteration of moving')
plt.ylabel('The best fitness value in these particles')
plt.show()