from xml.etree.ElementTree import QName
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
import time

def Rastrigin_fun(val, dim):
    obj = 0
    for dim_ite in range(dim): 
        obj += (val[:, dim_ite] ** 2 - 10 * np.cos(2 * np.pi * val[:, dim_ite]))
    obj += dim * 10
    return obj

def Styblinski_Tang_fun(val, dim):
    obj = 0
    for dim_ite in range(dim):
        obj += val[:, dim_ite] ** 4 - 16 * val[:, dim_ite] ** 2 + 5 * val[:, dim_ite]
    obj /= 2
    return obj

class NEW:
    ### build initial setting
    def __init__(self, fun, obj_sta, ori_num_peoples, ratio_scout, res_sup, res_back, res_rec, act_rng, ite, dim, LB, UB, apart_dis):
        self.fun = fun # 目標函數
        self.obj_sta = obj_sta # min or max
        self.ori_num_peoples = ori_num_peoples # 原始人數(族群不得低於此人數)
        self.ratio_scout = ratio_scout # 多少比例的人為斥侯
        self.res_sup = res_sup # 資源能夠支撐此族群活過幾個iteration
        self.res_back = res_back # 資源多久會回復
        self.res_rec = res_rec # 族群特別記錄表現好的幾個資源點
        self.act_rng = act_rng # 以據點為中心可活動範圍
        self.num_peoples_rec = [] # 紀錄人口增長狀況
        self.dim = dim # 維度
        self.LB = LB 
        self.UB = UB
        self.apart_dis = apart_dis # 兩駐紮點須隔距離

        self.center_rec = [] # 紀錄每次游牧中心點，讓游牧民族在資源還沒回復前不會回來
        self.next_center_rec = [] # 用以紀錄每次迭代中表現屬歷屆前幾佳結果(第一次陣列擺放值，第二個陣列放已經過了幾個iteration)
        self.next_center_rec_time = [] # 對照next_center_rec，紀錄距離這個最佳結果資源回復時間
        self.num_peoples = self.ori_num_peoples # 第一次游牧人數必為原先人數
        self.cen_loc = self.LB + np.subtract(self.UB, self.LB) * np.random.uniform(0, 1, size=(dim)) # random生成游牧中心據點
        self.center_rec.append(self.cen_loc) # 將此中心據點紀錄在陣列中
        self.loc = self.cen_loc + act_rng * np.subtract(self.UB, self.LB) * np.random.uniform(-0.5, 0.5, size=(self.num_peoples, dim)) # 決定好中心據點後，在據點附近生成其他游牧人民生活地點
        # 一旦座標超過上下界，則將該座標設為上下界
        for dim in range(self.dim):
            self.loc[self.loc[:, dim] < self.LB[dim], dim] = self.LB[dim]
            self.loc[self.loc[:, dim] > self.UB[dim], dim] = self.UB[dim]
        self.resouce = self.fun(self.loc, self.dim) # 計算本次游牧獲得資源量
        self.resouce_avg = [np.average(self.fun(self.loc, self.dim))] # 用來評比人口數上升或下降多少倍
        self.best_result_val = min(self.resouce) if obj_sta == 'min' else max(self.resouce) # 紀錄此次迭代最佳解
        self.best_result_val_ever = []
        self.best_result_val_ever.append(self.best_result_val.copy()) #紀錄至今表現最好結果的值
        self.best_result_loc = self.loc[np.argmin(self.resouce)] if obj_sta == 'min' else self.loc[np.argmax(self.resouce)] # 紀錄此次迭代最佳點位
        self.best_result_loc_ever = self.best_result_loc.copy() #紀錄至今表現最好結果的位置
        self.next_center_rec.append(self.best_result_loc) # 紀錄此次迭代最佳點位，用以當作未來是否再次回來此次游牧的依據
        self.next_center_rec_time.append(self.res_back)

    ### 找中心點(斥候找、和先前紀錄比較) ###
    def get_center_point(self):
        # 斥候找，同時確保斥候所找的這些位置都沒有在前幾輪已經游牧的範圍內
        cen_loc_cho = self.LB + np.subtract(self.UB, self.LB) * np.random.uniform(0, 1, size=(int(self.ratio_scout * self.num_peoples), dim)) # 讓游牧民族中的每個斥候在地圖中隨機尋找一點作為下次遷徙考量位置
        for cen in range(len(cen_loc_cho)):
            for use_cen in range(len(self.center_rec)):
                while True:
                    if distance.euclidean(cen_loc_cho[cen], self.center_rec[use_cen]) > self.apart_dis:
                        cen_loc_cho[cen] = self.LB + np.subtract(self.UB, self.LB) * np.random.uniform(0, 1, size=(dim))
                    else:
                        break
        # 找到斥候所找的位置中表現最好處
        find_loc_best = cen_loc_cho[np.argmin(self.fun(cen_loc_cho, self.dim))] if self.obj_sta == 'min' else cen_loc_cho[np.argmax(self.fun(cen_loc_cho, self.dim))]
        find_loc_index = np.argmin(self.fun(cen_loc_cho, self.dim)) if self.obj_sta == 'min' else np.argmax(self.fun(cen_loc_cho, self.dim))
        find_loc_val = self.fun(cen_loc_cho, self.dim)[find_loc_index]
        # 比較本次斥候找的點和歷屆紀錄最好的點
        sit = [val <= 0 for val in self.next_center_rec_time] # 判斷資源已回復點位
        if len(np.where(sit)[0]) != 0: # 如果此處資源已回復
            loc = self.next_center_rec[np.where(sit)[0][0]]
            new_loc = self.fun(np.expand_dims(np.array(loc), 0), self.dim) < find_loc_val if self.obj_sta == 'min' else self.fun0(np.expand_dims(np.array(loc), 0), self.dim) > find_loc_val
            if new_loc[0]:
                self.cen_loc = self.next_center_rec[np.where(sit)[0][0]]
        else:
            self.cen_loc = find_loc_best

    ### 根據中心點生成其他遊牧民眾位置 ###
    def get_other_point(self):
        self.loc = self.cen_loc + self.act_rng * np.subtract(self.UB, self.LB) * np.random.uniform(-0.5, 0.5, size=(self.num_peoples, self.dim)) # 決定好中心據點後，在據點附近生成其他游牧人民生活地點
        # 一旦座標超過上下界，則將該座標設為上下界
        for dim in range(self.dim):
            self.loc[self.loc[:, dim] < self.LB[dim], dim] = self.LB[dim]
            self.loc[self.loc[:, dim] > self.UB[dim], dim] = self.UB[dim]
        self.resouce = self.fun(self.loc, self.dim) # 計算本次游牧獲得資源量
        self.resouce_avg.append(np.average(self.fun(self.loc, self.dim)))
        self.best_result_val = min(self.resouce) if self.obj_sta == 'min' else max(self.resouce) # 紀錄此次迭代最佳解
        self.best_result_loc = self.loc[np.argmin(self.resouce)] if self.obj_sta == 'min' else self.loc[np.argmax(self.resouce)] # 紀錄此次迭代最佳點位
        # 將最佳解與位置資訊與歷史資訊做比較
        if self.obj_sta == 'min':
            self.best_result_val_ever.append(self.best_result_val.copy()) if self.best_result_val < self.best_result_val_ever[-1] else self.best_result_val_ever.append(self.best_result_val_ever[-1])  # 紀錄所有迭代最佳解
            self.best_result_loc_ever = self.best_result_loc.copy() if self.best_result_val < self.best_result_val_ever[-1] else self.best_result_loc_ever # 紀錄所有迭代最佳位置
        else:
            self.best_result_val_ever.append(self.best_result_val.copy()) if self.best_result_val > self.best_result_val_ever[-1] else self.best_result_val_ever.append(self.best_result_val_ever[-1]) # 紀錄所有迭代最佳解
            self.best_result_loc_ever = self.best_result_loc.copy() if self.best_result_val > self.best_result_val_ever[-1] else self.best_result_loc_ever # 紀錄所有迭代最佳位置

    ### 紀錄本次據點以及表現最佳之self.res_rec個據點位置，作為將來考量點
    def record_data(self):
        ### 紀錄本次據點至center_rec中，此陣列僅存放res_back個中心，目的在於讓此遊牧民族不要來這些地方駐紮 ###
        if len(self.center_rec) < self.res_back:
            self.center_rec.append(self.cen_loc)
        else:
            self.center_rec.pop(0)
            self.center_rec.append(self.cen_loc)

        ###  如果此次迭代發現之點位表現優於self.next_center_rec，便紀錄此點做為未來游牧考量 ###
        if len(self.next_center_rec) < 3: # 陣列紀錄點位未達3點
            val = self.fun(np.array(self.next_center_rec), self.dim) # 原本紀錄個點數值
            this_val = self.fun(np.expand_dims(self.best_result_loc,0), 2) # 本次游牧數值
            loc = np.where([this_val < ele for ele in val])[0] if self.obj_sta == 'min' else np.where([this_val > ele for ele in val])[0]
            if len(loc) != 0:
                self.next_center_rec.insert(loc[0], self.best_result_loc)
                self.next_center_rec_time.insert(loc[0], 6)
            self.next_center_rec_time = [ele - 1 for ele in self.next_center_rec_time]
        else:
            loc = np.where([self.best_result_val < ele for ele in self.fun(np.array(self.next_center_rec), self.dim)])[0] if self.obj_sta == 'min' else np.where([self.best_result_val > ele for ele in self.fun(np.array(self.next_center_rec), self.dim)])[0]
            if len(loc) != 0:
                # 確保這個好位置沒有在前幾輪就紀錄的點的範圍內
                for ind in range(loc[0], len(self.next_center_rec)):
                    # 如果在範圍內，就更新這個點
                    if distance.euclidean(self.best_result_val, self.next_center_rec[ind]) > self.apart_dis:
                        self.next_center_rec[ind] = self.best_result_loc
                        self.next_center_rec_time[ind] = 6
                        break
                    # 沒有在範圍內，就直接插入此點的位置
                    elif ind == len(self.next_center_rec) - 1:
                        self.next_center_rec.insert(loc[0], self.best_result_loc)
                        self.next_center_rec.pop(-1)
                        self.next_center_rec_time.insert(loc[0], 6)
                        self.next_center_rec_time.pop(-1)
            self.next_center_rec_time = [ele - 1 for ele in self.next_center_rec_time]

    ### 控制群體人口增減
    def control_people(self):
        # 計算人口成長比例
        if self.obj_sta == 'min':
            peo_grow_ratio = self.resouce_avg[-2] / self.resouce_avg[-1] - 1
        else:
            peo_grow_ratio = self.resouce_avg[-1] / self.resouce_avg[-2] - 1
        self.num_peoples += math.ceil(self.num_peoples * np.tanh(peo_grow_ratio))
        # 控制人口數量
        if self.num_peoples < self.ori_num_peoples:
            self.num_peoples = self.ori_num_peoples
        self.num_peoples_rec.append(self.num_peoples)

    ### 視覺化呈現
    def draw_result(self):
        # 畫出解收斂狀況
        plt.plot(self.best_result_val_ever)
        #plt.legend()
        plt.title('The convergence histories of algorithm')
        plt.xlabel('The iteration of moving')
        plt.ylabel('The best fitness value in these particles')
        plt.show()

        # 畫出人口變化狀況
        plt.plot(self.num_peoples_rec)
        #plt.legend()
        plt.title('The number of population change')
        plt.xlabel('The iteration of moving')
        plt.ylabel('The number of population')
        plt.show()

    ### 演算法執型執行流程
    def process(self, ite):
        st_time = time.time()
        for _ in range(ite):
            self.get_center_point()
            self.get_other_point()
            self.record_data()
            self.control_people()
        end_time = time.time()
        print(f'It totally use {end_time - st_time:.2f} seconds')
        print(f'The best location is {self.best_result_loc_ever}, and it\'s value is {self.best_result_val_ever[-1]:.5f}')
        self.draw_result()
        

### 參數設定，細節參照class
ori_num_peoples = 100
ratio_scout = 1/3
res_sup = 1
res_back = 6
res_rec = 3
act_rng = 0.05
obj_sta = 'min'
ite, dim = 100, 2
fun = Styblinski_Tang_fun
apart_dis = 10
LB, UB = [-5] * dim, [5] * dim

new_obj = NEW(fun, obj_sta, ori_num_peoples, ratio_scout, res_sup, res_back, res_rec, act_rng, ite, dim, LB, UB, apart_dis)
new_obj.process(ite)