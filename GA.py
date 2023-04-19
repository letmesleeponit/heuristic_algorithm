import numpy as np
import random
import bisect
import matplotlib.pyplot as plt
import math
import time

# 建立適應函數，依據不同適應函數將所求函數丟入物件
def obj_fun(val):
    x = val[0]
    y = val[1]
    output = - np.exp(-0.1 * np.add(x ** 2, y ** 2)) - np.exp(np.add(np.cos(2 * math.pi * x), np.cos(2 * math.pi * y)))
    return output

# 為使class具有更高的彈性，將非限制單一變數區間的限制，如 x + y <= 1等包含兩變數的限制式另外建立一函數丟入物件
def other_limit(decimal_arr, genes, dimen, num_gene, range_min, range_max):
    # 一旦該筆基因不符合 x + y <= 1 限制，便不斷替換該筆染色體基因
    while True:
        index_np = np.where(decimal_arr[0] + decimal_arr[1] > 1)[0] # 限制 x + y <= 1
        if len(index_np) != 0:
            # 更換不合染色體的基因
            for index in index_np:
                genes[index] = np.random.randint(0, 2, size=(num_gene))
            ini = 0
            temp_decimal_arr = []
            # 更新所有二進位染色體對應的十進位數值
            for index_dimen in range(len(dimen)):
                temp_genes = genes[:,ini:ini + dimen[index_dimen]]
                scalar_arr = [2 ** n for n in range(dimen[index_dimen])]
                temp_decimal_arr.append(range_min[index_dimen] + np.dot(temp_genes, scalar_arr) * (range_max[index_dimen] - range_min[index_dimen]) / (2 ** dimen[index_dimen] - 1))
                ini += dimen[index_dimen]
            decimal_arr = temp_decimal_arr
        else:
            break
        
    return decimal_arr, genes

class GA:
    ### 建立GA物件，包含以下參數
    '''
    num_ite: 染色體共有多少世代
    num_chro: 每代含有多少染色體
    num_gene: 每個染色體含有多少基因
    fun: 適應函數
    obj_sta: 所求為min或是max
    cross_PR: 染色體交配機率
    mutate_PR: 染色體變異機率
    range_min, range_max: 變數區間
    dimen: 變數共有多少個維度，以x、y來說便有兩個
    other_limit: 其他限制(限制多個變數間關係)
    num_chro_save: 用以儲存當代表現最好基因
    '''

    # 儲存各變數，並初始化所有基因
    def __init__(self, num_ite, num_chro, num_gene, fun, obj_sta, cross_PR, mutate_PR, range_min, range_max, dimen, other_limit, num_chro_save):
        self.num_ite =  num_ite
        self.genes = np.random.randint(0, 2, size=(num_chro, num_gene))
        self.fun = fun
        self.other_limit = other_limit
        self.obj_sta = obj_sta
        self.cross_PR = cross_PR
        self.mutate_PR = mutate_PR
        self.range_min = range_min
        self.range_max = range_max
        self.dimen = dimen
        self.num_chro_save = num_chro_save

    # 計算染色體所對應的十進位數值以及適應值
    def evaluation(self):
        num_chro = self.genes.shape[0]
        num_gene = self.genes.shape[1]
        ini = 0
        temp_decimal_arr = []
        # 計算各染色體對應十進位數值(須以for迴圈分別對不同維度資料，如x、y進行計算)
        for index_dimen in range(len(dimen)):
            temp_genes = self.genes[:,ini:ini + dimen[index_dimen]]
            scalar_arr = [2 ** n for n in range(dimen[index_dimen])]
            temp_decimal_arr.append(range_min[index_dimen] + np.dot(temp_genes, scalar_arr) * (range_max[index_dimen] - range_min[index_dimen]) / (2 ** dimen[index_dimen] - 1))
            ini += dimen[index_dimen]
        self.decimal_arr = temp_decimal_arr
        self.decimal_arr, self.genes = self.other_limit(self.decimal_arr, self.genes, self.dimen, num_gene, range_min, range_max)
        self.fit_arr = obj_fun(self.decimal_arr)
    
    # 選擇要進行交配的兩兩基因出來(使用輪盤式選擇)
    def selection(self):
        # 由於適應值有正有負，故先進行normalization，將數值控制在0~1之間，而為避免最小值沒有被選取機會(以及除以0問題)，故此處再加上0.5，
        # 使數值範圍介於0.5~1.5
        fit_arr_nor = (self.fit_arr - np.min(self.fit_arr)) / (np.max(self.fit_arr) - np.min(self.fit_arr)) + 0.5
        # 計算各染色體被選取機率
        if self.obj_sta == 'min':
            PR = (1 / abs(fit_arr_nor)) / sum(1 / abs(fit_arr_nor)) 
        elif self.obj_sta == 'max':
            PR = abs(fit_arr_nor) / sum(abs(fit_arr_nor))
        self.chrom_arr = [] # 放置兩兩交配染色體的index
        # 建立一index組成的array，並以sort語法找尋表現最佳的n個基因，方便之後儲存
        PR_index_arr = [x for x in range(len(PR))]
        self.chro_save = sorted(range(len(self.fit_arr)), key=lambda x: self.fit_arr[x])[0:self.num_chro_save] if self.obj_sta == 'min' else sorted(range(len(self.fit_arr)), key=lambda x: self.fit_arr[x])[-1 + self.num_chro_save:-1]
        
        # 根據先前計算的PR(機率)，不斷取出兩兩染色體出來進行交配
        for time in range(int((len(PR) - self.num_chro_save) / 2)):
            arr_index = []
            pro_wheel = [None] * len(PR)
            for i in range(len(PR), 0, -1):
                pro_wheel[i - 1] = np.sum(PR[0:i])
            ran_val = random.uniform(0, pro_wheel[-1])
            cho_index = bisect.bisect_left(pro_wheel, ran_val)
            arr_index.append(PR_index_arr[cho_index])
            PR = np.delete(PR, cho_index) # 不重複進行選取
            PR_index_arr = np.delete(PR_index_arr, cho_index) # 不重複進行選取

            pro_wheel = [None] * len(PR)
            for i in range(len(PR), 0, -1):
                pro_wheel[i - 1] = np.sum(PR[0:i])
            ran_val = random.uniform(0, pro_wheel[-1])
            cho_index = bisect.bisect_left(pro_wheel, ran_val)
            arr_index.append(PR_index_arr[cho_index])
            PR = np.delete(PR, cho_index) # 不重複進行選取
            PR_index_arr = np.delete(PR_index_arr, cho_index) # 不重複進行選取
            self.chrom_arr.append(arr_index)
            
    # 將染色體丟入其中，接著便會根據突變機率計算是否變異，並分別從不同維度(x、y)隨機選取位置進行突變。
    # 此處包成function，在每次進行完交配後將生成染色體丟入此function
    def mutation(self, temp_genes, counter, start_point, mutate_PR):
        for dimen in range(len(start_point)): # 控制不同維度
            for index in [counter, counter + 1]: # 控制不同染色體
                if random.uniform(0, 1) < mutate_PR:
                    rand_loc = int(random.uniform(0, 1) * self.dimen[dimen])
                    temp_genes[index][start_point[dimen] + rand_loc] = 0 if temp_genes[counter][start_point[dimen] + rand_loc] == 1 else 1

    # 將selection選取完的染色體兩兩進行交配          
    def crossover(self, cross_PR, mutate_PR):
        cross_arr = []
        temp_genes = np.full(self.genes.shape, 0) # 建立一空陣列，用以暫存每代生成之基因，最後再丟入self.genes
        # 根據交配機率計算是否予其進行交配
        for i in range(len(self.chrom_arr)):
            if random.uniform(0, 1) < cross_PR:
                cross_arr.append(i)
        self.chrom_arr = np.array(self.chrom_arr)[cross_arr] # 更新最後會進行交配的染色體

        # break point右側互換
        # 由於染色體由一個array構成，且此array可能包含不同維度資料(如10個bits，前4個為x，後6個為y)，
        # 故我們計算不同維度的起始和結束點，方便之後進行交配以及突變時控制區間
        start_point = list(reversed([sum(self.dimen[:i]) - self.dimen[i - 1] for i in range(len(self.dimen), 0, -1)]))
        end_point = [sum(self.dimen[:i + 1]) for i in range(len(self.dimen))]
        counter = 0 # 控制第n個染色體，將每次生成染色體丟入前面的temp_genes，儲存本世代生成基因
        for cross_group in self.chrom_arr:
            gene_break_point = np.add(start_point, [random.randint(0, self.dimen[i]) for i in range(len(dimen))]) # 計算要交換的斷點
            record = self.genes.copy()
            for i in range(len(self.dimen)): # 交配時，斷點右側的基因進行互換
                temp_genes[counter, start_point[i]:end_point[i]] = np.append(record[cross_group[0], start_point[i]:gene_break_point[i]], record[cross_group[1], gene_break_point[i]:end_point[i]])
                temp_genes[counter + 1, start_point[i]:end_point[i]] = np.append(record[cross_group[1], start_point[i]:gene_break_point[i]], record[cross_group[0], gene_break_point[i]:end_point[i]])
            self.mutation(temp_genes, counter, start_point, mutate_PR) # 將交配完生成之染色體丟入其中，計算是否要進行突變
            counter += 2
        temp_genes[counter : counter + len(self.chro_save)] = self.genes[self.chro_save] # 將每一代最優良的n筆基因進行儲存
        # 由於交配完染色體數目再加上原先儲存的優良染色體數量可能不足每代要求數目，
        # 因此我們再計算此次沒有被選入之其他表現較差基因的index
        # 接著從這些基因中隨機選取並填入下一代染色體列表，以滿足每代要求
        oth_index_chro = np.delete(np.array(range(0, self.genes.shape[0])), self.chro_save) 
        temp_genes[counter + len(self.chro_save) :] = self.genes[random.sample(list(oth_index_chro), k=self.genes.shape[0] - counter - len(self.chro_save) )]
        self.genes = temp_genes.copy()
    
    # 將上述GA演算法建立的每個功能串在一起，並根據所要求迭代次數進行計算
    # 步驟為: initialization、evaluation、selection、crossover、mutation
    def ite_running(self):
        self.evaluation()
        self.arr_point = []
        self.arr_result = []
        # 將初始化結果儲存在arr_point、arr_result中，最後視覺化呈現
        if self.obj_sta == 'min':
            self.arr_point.append([self.decimal_arr[i][np.argmin(self.fit_arr)] for i in range(len(dimen))])
            self.arr_result.append(self.fun(self.arr_point[-1]))
        else:
            self.arr_point.append([self.decimal_arr[i][np.argmax(self.fit_arr)] for i in range(len(dimen))])
            self.arr_result.append(self.fun(self.arr_point[-1]))
        # 根據要求世代數目進行迭代，並將每一代結果儲存在arr_point、arr_result中，最後視覺化呈現
        for _ in range(self.num_ite):
            self.selection()
            self.crossover(self.cross_PR, self.mutate_PR)
            self.evaluation()
            if self.obj_sta == 'min':
                self.arr_point.append([self.decimal_arr[i][np.argmin(self.fit_arr)] for i in range(len(dimen))])
                self.arr_result.append(self.fun(self.arr_point[-1]))
            else:
                self.arr_point.append([self.decimal_arr[i][np.argmax(self.fit_arr)] for i in range(len(dimen))])
                self.arr_result.append(self.fun(self.arr_point[-1]))

# 設定各種參數，建立物件，最後進行計算
num_ite = 100 # 要求世代數目
num_chro, num_gene = 100, 23 # 染色體和基因數目
num_chro_save = 10 # 要儲存的優良基因 
obj_sta = 'min' # 所求為min或max
cross_PR, mutate_PR = 0.9, 0.1 # 交配與變異機率
range_min, range_max = [-1, -2], [1, 1] # 變數區間(由兩個array組成，一個array裡面的值分別控制不同dimension)
dimen = [11, 12] # 控制不同dimension的數目(如x有11個，y有12個)


star_time = time.time()
obj = GA(num_ite, num_chro, num_gene, obj_fun, obj_sta, cross_PR, mutate_PR, range_min, range_max, dimen, other_limit, num_chro_save)
obj.ite_running()
end_time = time.time()
print(f'It totally use {end_time - star_time:.2f} seconds')

print(obj.arr_point[-1])
print(obj.arr_result[-1])
plt.plot(obj.arr_result)

plt.title('The convergence histories of GA')
plt.xlabel('The generation of chromosome')
plt.ylabel('The best fitness value in this generation')
plt.show()