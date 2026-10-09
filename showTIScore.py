import os
import sys

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from torch import mean, softmax
from skimage.measure import block_reduce
import numpy as np

from plt_many_graphs import get_d

def op_diff(n, data1, data2):
    if(n == 0):
        return data1
    elif(n == 1):
        return data2 - data1
    elif(n == 2):
        return 2 * (data2 - data1) / (data1 + data2)
    elif(n == 3):
        return 2 * (data2 - data1) / (mean(data1) + mean(data2))
    elif(n == 4):
        d = data2 - data1
        return d / abs(d).max()
    elif(n == 5):
        return norm_diff(data1, data2)

def op_norm(n, data):
    if(n == 0):
        return data / abs(data).sum()
    elif(n == 1):
        return (data - data.min()) / (data.max() - data.min())
    elif(n == 2):
        return softmax(data.flatten(), dim=0).reshape(data.shape)

def norm_diff(data1, data2, norm=1):
    # normalize each data with the same method (default: min-max) to align scales, then take the difference
    arr = op_norm(norm, np.vstack((data1, data2)))
    l = len(data1)
    return arr[l:] - arr[:l]

def save_image(save_dir, file_name):
    # create a new directory (if not exists) and save the current figure in it
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, file_name.replace(".", "-")))

diff = -1
diff_dict = {
    -1 : "None",
    0 : "raw1",
    1 : "normal_diff",
    2 : "relative_diff",
    3 : "average_diff",
    4 : "change_diff",
    5 : "norm_diff"
}
norm = 1
norm_dict = {
    -1 : "None",
    0: "l1",
    1: "min-max",
    2: "soft-max"
}
time = "1788760704.3488271"
model = "sret_s_no_loop"
data_path = f"./TaylorImportanceInLoop/{time}/{model}/data.pkl"
layer_dict = [
    'attn.qkv',
    'attn.proj',
    'mlp.fc1',
    'mlp.fc2'
]
# name_layer1 = 'transformers.1.blocks.0.attn.qkv'
# name_layer2 = 'transformers.1.blocks.2.attn.qkv'

name_layer1 = 'transformers.1.blocks.0.'
name_layer2 = 'transformers.1.blocks.8.'

pool_func = "max"
pool_func_dict = {
    "max": np.max,
    "mean": np.mean
}

pooling_size = (1, 1)

data = get_d(data_path)

# save directory (name without layer name such as attn-proj)
save_dir = f"./TaylorImportanceInLoop/{time}/{model}/" + f"{diff_dict[diff]}_{norm_dict[norm]}_{name_layer1}_{name_layer2}_poolsize{pooling_size}_pool-func_{pool_func}".replace(".", "-")

for layer in layer_dict:
    data1 = data["TaylorImportance"][(name_layer1 + layer, 'weight')]
    data2 = data["TaylorImportance"][(name_layer2 + layer, 'weight')]

    data1 = data1.detach().cpu().numpy()
    data2 = data2.detach().cpu().numpy()

    if(data1.shape != data2.shape):
        sys.exit(1)

    cell_size = 0.04
    nrow, ncol = data1.shape
    plt.subplots(figsize=(ncol * cell_size + 1.5, nrow * cell_size + 1.5), constrained_layout=True)

    if(diff > -1):
        d = op_diff(diff, data1, data2)
        # dl = len(d)//3
        # d = d[dl:dl*2]

        # pooling (np.max or np.average)
        d_pool = block_reduce(d, pooling_size, func=pool_func_dict[pool_func])

        # max value for imshow()
        m, mi = abs(d_pool.max()), abs(d_pool.min())
        if m < mi :
            m = mi

        plt.imshow(
            d_pool, 
            cmap="bwr",
            norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
        )
        layer_title = f"{name_layer2 + layer} - {name_layer1 + layer}"

    elif(norm > -1):
        d = op_norm(norm, data1)

        # pooling (np.max or np.average)
        d_pool = block_reduce(d, pooling_size, func=np.max)

        plt.imshow(
            d_pool, 
            cmap="Greys",
            vmin=0,
            vmax=1
        )
        layer_title = f"{name_layer1 + layer}"

    print(layer)
    print(data1.mean())
    print(data2.mean())
    print(d.max())
    print(d.min())

    plt.colorbar()
    # plt.suptitle(f"Taylor Importance Variation In Loop", fontweight="bold")
    plt.title(f"Taylor Importance Variation In Loop\n{time} {model} {diff_dict[diff]} {norm_dict[norm]}\n{layer_title}\nmax = {d.max():.8}, min = {d.min():.8}\n_poolsize{pooling_size}_pool-func_{pool_func}")
    save_image(save_dir, layer)

# plt.show()
