import torch

import SReT

model = SReT.SReT_S(pretrained=False)
model.load_state_dict(torch.load('./pre-trained/SReT_S.pth')['model'])\

# model = timm.create_model(
#     "vit_base_patch16_224",
#     pretrained=True,
# )




print(model)
