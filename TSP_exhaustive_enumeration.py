import pandas as pd
import numpy as np
from itertools import permutations
import time
from typing import List

# 計算不同城市間移動總距離
def get_the_val(st_array:List[int], dis_mat:pd.DataFrame) -> int:
    total_dis = 0
    for ind in range(0, len(st_array)):
        total_dis += dis_mat.iloc[st_array[ind] - 1, st_array[ind + 1] - 1]
        if ind == len(st_array) - 2:
            total_dis += dis_mat.iloc[st_array[ind+1] - 1, st_array[0] - 1]
            break

    return total_dis

### 讀取資料(距離矩陣)
TSP_file_path = r'./SA TS Problems.xlsx' # TSP問題的檔案位置
TSP_file_sheet = 'Q1' # TSP問題的距離矩陣放在哪個工作表
dis_mat = pd.read_excel(TSP_file_path, sheet_name=TSP_file_sheet, index_col=0, header=0)
        
### 暴力求解
count = 0
best_val = 100000
best_loc = ''
all_combi = list(permutations(range(1,11), 10)) # 獲取所有組合
st_time = time.time()
for combi in all_combi:
    count += 1
    val = get_the_val(np.array(combi), dis_mat)
    if val < best_val:
        best_val, best_loc = val, combi
end_time = time.time()
print('Total distance in traveling across the city:', best_val)
print('Best route:', best_loc)
print('Total cost:', end_time - st_time, 's')





