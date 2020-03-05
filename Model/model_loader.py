import torch
from Model import base_model


def save_checkpoint(args, model, path):
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'other_infos': None
    }
    torch.save(state, path)


def load_checkpoint(path):
    state = torch.load(path)
    args, loaded_state_dict = state['args'], state['state_dict']
    model = base_model.SimpleModel(args)
    # 读取参数并更新
    model_state_dict = model.state_dict()
    model_state_dict.update(loaded_state_dict)
    model.load_state_dict(model_state_dict)
    return model
