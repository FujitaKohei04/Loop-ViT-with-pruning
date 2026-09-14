import sys

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from torch import mean

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

diff = 1
diff_dict = {
    0 : "raw1",
    1 : "normal_diff",
    2 : "relative_diff",
    3 : "average_diff",
    4 : "change_diff"
}
time = "1788760704.3488271"
model = "sret_s_no_loop"
data_path = f"./TaylorImportanceInLoop/{time}/{model}/data.pkl"
name_layer1 = 'transformers.1.blocks.0.mlp.fc2'
name_layer2 = 'transformers.1.blocks.2.mlp.fc2'

data = get_d(data_path)
data1 = data["TaylorImportance"][(name_layer1, 'weight')]
data2 = data["TaylorImportance"][(name_layer2, 'weight')]

if(data1.shape != data2.shape):
    sys.exit(1)

d = op_diff(diff, data1, data2).detach().cpu().numpy()

plt.subplots(figsize=(10,10))

# max value for imshow()
m, mi = abs(d.max()), abs(d.min())
if m < mi :
    m = mi

plt.imshow(
    d, 
    cmap="bwr",
    norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-1*m, vmax=m)
    )

print(mean(data1))
print(mean(data2))
print(d.max())
print(d.min())

plt.colorbar()
plt.title(f"Taylor Importance Variation In Loop\n{time} {model} {diff_dict[diff]}\n{name_layer2} - {name_layer1}\nmax = {d.max():.8}, min = {d.min():.8}")
plt.savefig(f"./TaylorImportanceInLoop/{time}/{model}/"+f"{diff_dict[diff]}_{name_layer1}_{name_layer2}".replace(".", "-"))
plt.show()
