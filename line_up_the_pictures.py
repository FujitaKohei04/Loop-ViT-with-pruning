from pathlib import Path
from PIL import Image

# 設定
dir_path = Path("./analysis-data/activations/sret_s")  # 対象のディレクトリ
thumb_size = (200, 200)  # 各画像のサイズ (幅, 高さ)
cols = 5  # 1行に並べる枚数

# 画像ファイルの取得
valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
img_paths = [p for p in dir_path.iterdir() if p.suffix.lower() in valid_exts]

if not img_paths:
    print("画像が見つかりませんでした。")
    exit()

# 枚数から全体の行数とキャンバスサイズを計算
num_imgs = len(img_paths)
rows = (num_imgs + cols - 1) // cols
canvas_width = cols * thumb_size[0]
canvas_height = rows * thumb_size[1]

# 白地の大きなキャンバスを作成
canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

# 画像を1枚ずつリサイズして貼り付け
for i, path in enumerate(img_paths):
    with Image.open(path) as img:
        img_resized = img.resize(thumb_size)

        col = i % cols
        row = i // cols
        x = col * thumb_size[0]
        y = row * thumb_size[1]

        canvas.paste(img_resized, (x, y))

# 保存
canvas.save("combined_image.jpg")
print("1枚の画像に結合して保存しました！")