import torch

import SReT
import SReT_no_loop
import cp_pre_w

model = SReT.SReT_S(pretrained=False)
model.load_state_dict(torch.load('./pre-trained/SReT_S.pth')['model'])

model2 = cp_pre_w.load_SReT_S_no_loop()

# model = timm.create_model(
#     "vit_base_patch16_224",
#     pretrained=True,
# )



# print(sum(p.numel() for p in model.parameters()))
# print(sum(p.numel() for p in model2.parameters()))
print(model)