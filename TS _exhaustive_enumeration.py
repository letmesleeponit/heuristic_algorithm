import pandas as pd
import numpy as np
from itertools import permutations
import time

# 單純計算距離
def get_the_val(st_array, dis_mat):
    total_dis = 0
    for ind in range(0, len(st_array)):
        total_dis += dis_mat.iloc[st_array[ind] - 1, st_array[ind + 1] - 1]
        if ind == len(st_array) - 2:
            total_dis += dis_mat.iloc[st_array[ind+1] - 1, st_array[0] - 1]
            break

    return total_dis

### 獲取距離矩陣
TSP_file_path = r'C:\Users\mljgs\OneDrive\桌面\Jerry\110-2\課程\計算智慧\作業\TS\SA TS Problems.xlsx' # TSP問題的檔案位置
TSP_file_sheet = 'Q1' # TSP問題的距離矩陣放在哪個工作表
dis_mat = pd.read_excel(TSP_file_path, sheet_name=TSP_file_sheet, index_col=0, header=0)
        
### 暴力求解
count = 0
best_val = 100000
best_loc = ''
all_combi = list(permutations(range(1,11), 10)) # 獲取所有組合
sd_time = time.time()
for combi in all_combi:
    count += 1
    val = get_the_val(np.array(combi), dis_mat)
    if val < best_val:
        best_val = val
        best_loc = combi
end_time = time.time()
print(best_val)
print(best_loc)
print(end_time - sd_time)





