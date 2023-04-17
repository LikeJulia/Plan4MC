import os
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

# theme
plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')

# title
# title = 'Harvest double_plant Average'
save_name = 'flower'
save_path = 'paper_fig'

xname = 'Epochs'
yname = 'Average Success Rate'

# num of data
num_horizon = 800
x = list(range(num_horizon))

# Motion clip
f1 = 'motionclip/flower/1.txt'
f2 = 'motionclip/flower/2.txt'
f3 = 'motionclip/flower/3.txt'
f4 = 'motionclip/flower/4.txt'
f5 = 'motionclip/flower/5.txt'

# Motion clip simple
f6 = 'motionclip_simple/flower/1.txt'
f7 = 'motionclip_simple/flower/2.txt'
f8 = 'motionclip_simple/flower/3.txt'
f9 = 'motionclip_simple/flower/4.txt'
f10 = 'motionclip_simple/flower/5.txt'

# MineCLIP(pre-trained)
f11 = 'data/2-base_harvest_1_double_plant/2-base_harvest_1_double_plant_s7/progress.txt'
f12 = 'data/base_harvest_1_double_plant/base_harvest_1_double_plant_s7/progress.txt'

# MineCLIP(ablation)
f16 = 'data/2-ziluo_harvest_1_double_plant/2-ziluo_harvest_1_double_plant_s7/progress.txt'
f17 = 'data/ziluo_harvest_1_double_plant/ziluo_harvest_1_double_plant_s7/progress.txt'



group = [[f1, f2, f3, f4, f5],
        [f6, f7, f8, f9, f10],
        [f12, f11],
        [f16, f17]]

label = ['CLIP4MC', # ours v2
        'CLIP4MC-simple', # ours v1
        'MineCLIP(pre-trained)', # baseline
        'MineCLIP(scratch)', # ablation
        ]

# smooth the curves
def smooth(arr, weight=0.98): #weight是平滑度，tensorboard 默认0.6
    last = 0 # last = arr[0]
    smoothed = []
    for point in arr:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# i = 0
# # Initialize an empty list to store moving averages
# moving_averages = []
  
# # Loop through the array to consider
# # every window of size 3
# while i < len(arr) - window_size + 1:
    
#     # Store elements from i to i+window_size
#     # in list to get the current window
#     window = arr[i : i + window_size]
  
#     # Calculate the average of current window
#     window_average = round(sum(window) / window_size, 2)
      
#     # Store the average of current
#     # window in moving average list
#     moving_averages.append(window_average)
      
#     # Shift window to right by one position
#     i += 1

# initialize data array
alldata = []
for i in range(len(group)):
    alldata.append([])
    for j in range(len(group[i])):
        with open(group[i][j], 'r') as f:
            lines = f.read().splitlines()

        for k,l in enumerate(lines):
            lines[k] = l.split('\t')

        i_y = lines[0].index('AverageEpSuccess')
        lines = np.array(lines)
        y = np.array(lines[1:num_horizon+1, i_y])
        
        y = y.astype(np.float)
        alldata[i].append(smooth(y))
    alldata[i] = np.array(alldata[i])


for i in range(len(alldata)):
    color=palette(i)#算法1颜色
    avg=np.mean(alldata[i],axis=0)
    std=np.std(alldata[i],axis=0)
    
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std/2))) #上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std/2))) #下方差
    plt.plot(x, avg, color=color,label=label[i],linewidth=3.0)
    plt.fill_between(x, r1, r2, color=color, alpha=0.2)


if not os.path.exists(save_path):
    os.mkdir(save_path)

plt.legend(loc='lower right')
plt.xlabel(xname)
plt.ylabel(yname)


plt.savefig(os.path.join(save_path, save_name+'.pdf'))
plt.savefig(os.path.join(save_path, save_name+'.png'))