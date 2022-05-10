import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time

class TS:

    ### 計算移動總距離
    '''
    comb: 控制是否要進行交換組合的參數
    '''
    def get_the_val(self, comb):
        # 先換組合再計算距離
        if comb != '':
            total_dis = np.zeros(len(comb))
            for comb_ind in range(len(comb)):
                # 交換兩元素
                temp_st_array = self.st_array.copy()
                ind_one, ind_sec = self.st_array.index(comb[comb_ind][0]), self.st_array.index(comb[comb_ind][1])
                val_one, val_sec = self.st_array[ind_one], self.st_array[ind_sec]
                temp_st_array[ind_one], temp_st_array[ind_sec] = val_sec, val_one
                # 計算交換後距離
                for ind in range(0, len(temp_st_array)):
                    total_dis[comb_ind] += self.dis_mat.iloc[temp_st_array[ind] - 1, temp_st_array[ind + 1] - 1]
                    if ind == len(temp_st_array) - 2:
                        total_dis[comb_ind] += self.dis_mat.iloc[temp_st_array[ind+1] - 1, temp_st_array[0] - 1]
                        break
        # 單純計算距離
        else:
            total_dis = 0
            for ind in range(0, len(self.st_array)):
                total_dis += self.dis_mat.iloc[self.st_array[ind] - 1, self.st_array[ind + 1] - 1]
                if ind == len(self.st_array) - 2:
                    total_dis += self.dis_mat.iloc[self.st_array[ind+1] - 1, self.st_array[0] - 1]
                    break
        return total_dis

    ### 參數設定，解釋參照最下方建立物件前註解
    def __init__(self, tenure, aspitarion_cri, moves, TSP_file_path, TSP_file_sheet, obj_sta, ite):
        self.tenure = tenure
        self.aspitarion_cri =  aspitarion_cri
        self.moves = moves
        self.obj_sta = obj_sta
        self.ite = ite

        # 讀取TSP問題之距離矩陣
        self.dis_mat = pd.read_excel(TSP_file_path, sheet_name=TSP_file_sheet, index_col=0, header=0)
        # 建立tabu search計算用之矩陣
        self.tabu_mat = np.zeros((len(self.dis_mat),len(self.dis_mat)))
        # 建立初始排序
        self.st_array = [num for num in range(1, len(self.dis_mat) + 1)]
        # 計算原始組合之fit value
        self.ori_fit_val = self.get_the_val('')
        self.best_val = self.ori_fit_val.copy() # 紀錄目前為止最佳解
        self.best_combi = self.st_array.copy() # 紀錄目前為止最佳解之組合
        self.best_val_rec = [] # 紀錄每次迭代所得到最佳解

    ### 一旦沒有在tenure中，又或者超出aspitarion_cri，進行兩解交換  
    def progress(self):
        comb = [random.sample(range(1,len(self.dis_mat) + 1), 2) for _ in range(self.moves)] # 隨機生成self.moves個解
        fit_val = self.get_the_val(comb) # 計算最應解所得fit value
        change_ind = np.argsort(fit_val)
        for ind in range(0,len(change_ind)):
            ind = len(change_ind) - ind if self.obj_sta == 'max' else ind
            loc_one = sorted(comb[change_ind[ind]])[0]
            loc_sec = sorted(comb[change_ind[ind]])[1]
            # 條件一，沒有在tenure中
            con_one = self.tabu_mat[loc_one-1, loc_sec-1] == 0 
            # 條件二，超出aspitarion criterion
            if self.obj_sta == 'min':
                con_sec = fit_val[change_ind[ind]] - self.ori_fit_val < self.aspitarion_cri  
            else:
                con_sec = fit_val[change_ind[ind]] - self.ori_fit_val > self.aspitarion_cri       
            # 滿足兩條件其中之一便進行交換
            if  con_one or con_sec:
                ind_one, ind_sec = self.st_array.index(loc_one), self.st_array.index(loc_sec)
                val_one, val_sec = self.st_array[ind_one], self.st_array[ind_sec]
                self.st_array[ind_one], self.st_array[ind_sec] = val_sec, val_one
                ### 更新tabu search matrix
                # 等待次數-1
                for i in range(len(self.tabu_mat)):
                    for j in range(i, len(self.tabu_mat)):
                        self.tabu_mat[i, j] = self.tabu_mat[i, j] - 1 if self.tabu_mat[i, j] > 0 else self.tabu_mat[i, j]
                self.tabu_mat[loc_sec-1, loc_one-1] += 1 # 交換次數 + 1
                self.tabu_mat[loc_one-1, loc_sec-1] = self.tenure
                break # 結束本次iteration交換

    ### 紀錄每次迭代組合結果
    def record(self):
        self.ori_fit_val = self.get_the_val('') # 計算目前組合之fit value
        if self.obj_sta == 'min' and self.ori_fit_val < self.best_val:
            self.best_val = self.ori_fit_val.copy()
            self.best_combi = self.st_array.copy()
        elif self.obj_sta == 'max' and self.ori_fit_val > self.best_val:
            self.best_val = self.ori_fit_val.copy()
            self.best_combi = self.st_array.copy()
        self.best_val_rec.append(self.best_val)

    ### 視覺化呈現
    def draw_result(self):
        # 畫出解收斂狀況
        plt.plot(self.best_val_rec)
        plt.title('The convergence histories of algorithm')
        plt.xlabel('The iteration of moving')
        plt.ylabel('The best fitness value in these particles')
        plt.show()

    ### 演算法流程
    def pipeline(self):
        st_time = time.time()
        for _ in range(0,self.ite):
            self.progress()
            self.record()
        end_time = time.time()
        self.draw_result()
        print(f'The route is {self.best_combi}, and it total need {self.best_val} time')
        print(f'It totally use {end_time - st_time} seconds')
        print(pd.DataFrame(self.tabu_mat.astype(np.int64)))

### parameter setting
tenure = 4 # 短期記憶4次內的交換
aspitarion_cri = 10 # 解一旦進步超過此參數，便允許進行交換，不理會限制
moves = 10 # 每次迭代生成這些數量的可行解
TSP_file_path = r'C:\Users\mljgs\OneDrive\桌面\Jerry\110-2\課程\計算智慧\作業\TS\SA TS Problems.xlsx' # TSP問題的檔案位置
TSP_file_sheet = 'Q1' # TSP問題的距離矩陣放在哪個工作表
obj_sta = 'min' # 所求解為max or min
ite = 100 # 迭代次數

TS_obj = TS(tenure, aspitarion_cri, moves, TSP_file_path, TSP_file_sheet, obj_sta, ite)
TS_obj.pipeline()


