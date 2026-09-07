import torch
pth_path = "./pre-trained/SReT_S.pth"

cp = torch.load(pth_path, map_location="cpu")
print(str(cp["model"].keys()).replace(",", "\n"))