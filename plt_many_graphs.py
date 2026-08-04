import matplotlib.pyplot as plt
import numpy as np

# データの設定
data1 = np.load("./prun_acc_figure/sret_s/data.npy")
data2 = np.load("./prun_acc_figure/vit_base_patch16_224/data.npy")
x1 = data1[0][0]
x2 = data2[0][0]
y1 = data1[2][0]  # データ1
y2 = data2[2][0]  # データ2

plt.figure(figsize=(12, 4))
plt.ticklabel_format(style='plain', useOffset=False, axis='x')

# 1本目の折れ線グラフ
plt.plot(x1, y1, label="SReT_S", marker="o", color="blue")

# 2本目の折れ線グラフ（上から重ねる）
plt.plot(x2, y2, label="ViT Base patch16 224", marker="s", color="orange")


# グラフのカスタマイズ
plt.xlabel("parameter")
plt.ylabel("accuracy")
plt.title("ImageNet 100 Top 5 Acc")
plt.legend()  # 凡例（どれがどのラインか）を表示
plt.grid(True)  # グリッド線を表示

# 保存・表示
# plt.savefig("stacked_line_chart.png", bbox_inches="tight")
plt.show()