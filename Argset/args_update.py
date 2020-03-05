import time

# 根据数据更新参数


def update(args, example_data):
    args.model_path = args.model_path + \
        time.strftime('-%m-%d-%H-%M', time.localtime())
    feat_size = example_data.antibody_dgl.ndata['feature']
    args.gcn_input_size = feat_size.shape[-1]
    return args
