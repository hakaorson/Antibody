from Model import data_loader
from Argset import global_args
from Model import trainer
from Model import base_model
import torch
import random
'''
torch.manual_seed(666)
random.seed(666)
'''


def arg_update(args, example):
    feat_size = example.antibody.ndata['feature']
    args.gcn_input_size = feat_size.shape[-1]
    return args


def main():
    args = global_args.get_model_args()
    all_data = data_loader.StructData(args)
    args = arg_update(args, all_data.train_data[0])
    model = base_model.SimpleModel(args)
    for epoch in range(args.epoch):
        train_data_gener = data_loader.BatchGenerator(
            all_data.train_data, args.batch_size)
        model = trainer.train(args, model, train_data_gener)
        pass
    pass


if __name__ == '__main__':
    main()
