import sys

from matplotlib import pyplot as plt

from plt_many_graphs import get_d

diff = 1
diff_dict = {
    1 : "normal_diff",
    2 : "relative_diff"
}
time = "1788760704.3488271"
model = "sret_s_no_loop"
data_path = f"./TaylorImportanceInLoop/{time}/{model}/data.pkl"
name_layer1 = 'transformers.0.blocks.0.attn.qkv'
name_layer2 = 'transformers.0.blocks.2.attn.qkv'

def op_diff(n, data1, data2):
    if(n == 1):
        return abs(data1 - data2)
    elif(n == 2):
        return 2 * abs(data1 - data2) / (data1 + data2)

data = get_d(data_path)
data1 = data["TaylorImportance"][(name_layer1, 'weight')]
data2 = data["TaylorImportance"][(name_layer2, 'weight')]

if(data1.shape != data2.shape):
    sys.exit(1)



plt.subplots(figsize=(7,9))
plt.imshow(
    op_diff(diff, data1, data2).detach().cpu().numpy(), 
    cmap="viridis",
    )

plt.colorbar()
plt.title(f"Taylor Importance Variation In Loop\n{time} {model} {diff_dict[diff]}\n{name_layer1} - {name_layer2}")
plt.savefig(f"./TaylorImportanceInLoop/{time}/{model}/"+f"{diff_dict[diff]}_{name_layer1}_{name_layer2}".replace(".", "-"))
plt.show()
