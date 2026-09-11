import sys

from matplotlib import pyplot as plt

from plt_many_graphs import get_d

time = "1788760704.3488271"
model = "sret_s_no_loop"
data_path = f"./TaylorImportanceInLoop/{time}/{model}/data.pkl"
name_layer1 = 'transformers.0.blocks.0.attn.qkv'
name_layer2 = 'transformers.0.blocks.2.attn.qkv'

data = get_d(data_path)
data1 = data["TaylorImportance"][(name_layer1, 'weight')]
data2 = data["TaylorImportance"][(name_layer2, 'weight')]

if(data1.shape != data2.shape):
    sys.exit(1)

plt.figure(figsize=(12,8))
plt.ylim(0, 0.01)

y = data1.detach().cpu().numpy().flatten()
x = range(len(y))

plt.bar(
    x,
    y,
)

plt.grid(True, linestyle='--')
plt.show()