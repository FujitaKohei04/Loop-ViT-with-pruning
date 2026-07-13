import timm

model = timm.create_model(
    "vit_base_patch16_224"
)
print(model)