import os
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

x_l = "parameter" # pruning_rate, parameter 
y_l = "imagenet_top1" # imagenet_top1, imagenet_top5

def get_d(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

# get data
data = {}
for sdir in Path("./prun_acc_figure").iterdir():
    if sdir.is_dir():
        data[sdir.name] = get_d(os.path.join(sdir, "data.pkl"))
data = {k: data[k] for k in [
    "sret_s", "sret_lt", "sret_t", 
    "vit_base", "vit_small", "vit_tiny"
    ] if k in data}

# plt setting
plt.figure(figsize=(18, 9))
plt.rcParams['font.size'] = 25
plt.ticklabel_format(useOffset=False, axis='x')
plt.title(f"compare models by {"remaining parameter rate" if x_l == "pruning_rate" else x_l}")
plt.grid(True)
plt.ylim(0, 1.0)
plt.xlim(0, 6000000)



# plt the data
for model, d in data.items():
    plt.xlabel("remaining parameter rate" if x_l == "pruning_rate" else x_l )
    plt.ylabel(y_l)
    # if x_l == "parameter":
        # plt.xlim(0, 35000000)
    plt.plot(
        1.0 - np.array(d["x"][x_l]) if x_l == "pruning_rate" else d["x"][x_l],
        d["y"][y_l],
        label=model,
        marker=("o" if "sret_s" in model or "vit_base" in model
            else "^" if "sret_lt" in model or "vit_small" in model
            else "x"),
        # ls=("solid" if "sret_s" in model or "vit_base" in model
        #     else "dashed" if "sret_lt" in model or "vit_small" in model
        #     else "dashdot"),
        color=(
            "#e44" if "sret_s" in model 
            else "#e88" if "sret_lt" in model
            else "#fbb" if "sret_t" in model
            else "#48a" if "vit_base" in model
            else "#6ac" if "vit_small" in model
            else "#8bd"

        ) 
        
        
    )

plt.legend()
plt.savefig(f"./pru_acc_all_model_figure/graph_{x_l}_and_{y_l}_xmax1e7")
plt.show()