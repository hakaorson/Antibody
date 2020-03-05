# 根据数据更新参数
def update(args, example_data):
    feat_size = example_data.antibody_dgl.ndata['feature']
    args.gcn_input_size = feat_size.shape[-1]
    return args
