from torch import nn

def collect_weights(model):
    weights = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weights[name] = module.weight.detach().cpu()

    return weights