import matplotlib.pyplot as plt
import numpy as np

# データの設定
data1 = np.load("./prun_acc_figure/sret_s/data.npy")
data2 = np.load("./prun_acc_figure/vit_base_patch16_224/data.npy")
x = data1[0][0]
y1 = data1[2][0]  # データ1
y2 = data2[2][0]  # データ2

# 1本目の折れ線グラフ
plt.plot(x, y1, label="SReT_S", marker="o", color="blue")

# 2本目の折れ線グラフ（上から重ねる）
plt.plot(x, y2, label="ViT Base patch16 224", marker="s", color="orange")

# グラフのカスタマイズ
plt.xlabel("parameter")
plt.ylabel("accuracy")
plt.title("sret and vit")
plt.legend()  # 凡例（どれがどのラインか）を表示
plt.grid(True)  # グリッド線を表示

# 保存・表示
# plt.savefig("stacked_line_chart.png", bbox_inches="tight")
plt.show()