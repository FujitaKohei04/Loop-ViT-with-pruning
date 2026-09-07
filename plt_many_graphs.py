import os
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

x_l = "pruning_rate" # pruning_rate, parameter 
y_l = "imagenet_top1" # imagenet_top1, imagenet_top5
plt_all = False
plt_model = [
    "sret_s",
    "sret_s_no_loop",
]
# "sret_lt",
# "sret_s",
# "sret_s_no_loop",
# "sret_t",
# "vit_base_patch16_224",
# "vit_small_patch16_224",
# "vit_tiny_patch16_224",

def get_d(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def plt_data(model, d):
    plt.xlabel(x_l)
    plt.ylabel(y_l)
    plt.plot(
        d["x"][x_l],
        d["y"][y_l],
        label=model,
        marker="o"
    )
    
# get data
data = {}
for sdir in Path("./prun_acc_figure").iterdir():
    if sdir.is_dir():
        data[sdir.name] = get_d(os.path.join(sdir, "data.pkl"))

# plt setting
plt.figure(figsize=(18, 6))
plt.ticklabel_format(style='plain', useOffset=False, axis='x')
plt.title(f"compare models by {x_l}")
plt.grid(True)
# plt.xlim(0, 6000000)

# plt all data
if(plt_all == True) :
    for model, d in data.items():
        plt_data(model, d)
else:
    for m in plt_model:
        plt_data(m, data[m])

plt.legend()
plt.savefig(f"./pru_acc_all_model_figure/graph {x_l} and {y_l}")
plt.show()