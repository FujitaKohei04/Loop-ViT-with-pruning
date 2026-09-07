#if you have same structure of model, can run.
import torch

import SReT
import SReT_no_loop
from main_measure import set_seed


def copy_pretrained_weights(old_model, new_model):
    for old_transformer, new_transformer in zip(
        loop_model.transformers,
        no_loop_model.transformers
    ):
        for old_block, new_block in zip(
            old_transformer.blocks,
            new_transformer.blocks
        ):
            new_block.load_state_dict(old_block.state_dict())

def check_same_weights(old_model, new_model):
    # loop版とno-loop版の実行順で比較
    for i, (old_blk, new_blk) in enumerate(
        zip(old_model.transformers[0].blocks,
            new_model.transformers[0].blocks)
    ):
        same = all(
            torch.equal(old_param, new_param)
            for old_param, new_param
            in zip(old_blk.state_dict().values(),
                new_blk.state_dict().values())
        )
        print(i, same)
    for i, (old_blk, new_blk) in enumerate(
            zip(old_model.transformers[1].blocks,
                new_model.transformers[1].blocks)
        ):
            same = all(
                torch.equal(old_param, new_param)
                for old_param, new_param
                in zip(old_blk.state_dict().values(),
                    new_blk.state_dict().values())
            )
            print(i, same)
    for i, (old_blk, new_blk) in enumerate(
            zip(old_model.transformers[2].blocks,
                new_model.transformers[2].blocks)
        ):
            same = all(
                torch.equal(old_param, new_param)
                for old_param, new_param
                in zip(old_blk.state_dict().values(),
                    new_blk.state_dict().values())
            )
            print(i, same)
    

    # check same weights
    print("independe?")
    print(torch.equal(
        new_model.transformers[0].blocks[2].attn.qkv.weight,
        new_model.transformers[0].blocks[6].attn.qkv.weight
    ))

    # check that they use different memory.
    print(
        new_model.transformers[0].blocks[2].attn.qkv.weight.data_ptr()
        ==
        new_model.transformers[0].blocks[6].attn.qkv.weight.data_ptr()
    )

def check(nm):
    #load and setting
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    om = SReT.SReT_S(pretrained=False)
    om.load_state_dict(torch.load('./pre-trained/SReT_S.pth')['model'])

    #cp weights of old model to nm
    # copy_pretrained_weights(om, nm)

    #confirm weather or not it's same weights
    check_same_weights(om, nm)

def load_SReT_S_no_loop():
    nm = SReT_no_loop.SReT_S_no_loop(pretrained=False)
    nm.load_state_dict(torch.load('./pre-trained/SReT_S.pth')['model'])

    # check(nm)

    return nm