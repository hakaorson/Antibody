from Model import data_loader
from Argset import global_args
from Model import trainer
from Model import base_model
import numpy as np
import torch
torch.manual_seed(1)
np.random.seed(1)


def arg_update(args, example):
    feat_size = example.antibody_dgl.ndata['feature']
    args.gcn_input_size = feat_size.shape[-1]
    return args


def get_infos(all_graphs):
    antibody_infos, antigen_infos = [], []
    for item in all_graphs:
        antibody, antigen = item[0][0], item[1][0]
        antibody_label = antibody.ndata['label'].detach().numpy()
        antigen_label = antigen.ndata['label'].detach().numpy()
        antibody_infos.append([sum(antibody_label), len(
            antibody_label), antibody.number_of_edges()])
        antigen_infos.append([sum(antigen_label), len(
            antigen_label), antigen.number_of_edges()])
    antibody_infos = np.sum(np.array(antibody_infos), axis=0)
    antigen_infos = np.sum(np.array(antigen_infos), axis=0)
    print(antibody_infos)
    print(antigen_infos)


def main():
    args = global_args.get_model_args()
    all_data = data_loader.StructData(args)
    get_infos(all_data.pos_data)
    args = arg_update(args, all_data.train_data[0])
    model = base_model.SimpleModel(args)
    for epoch in range(args.epoch):
        # print('epoch:', epoch)
        train_data_gener = data_loader.BatchGenerator(
            all_data.pos_stuct_data, args.batch_size)
        model = trainer.train(args, model, train_data_gener)


if __name__ == '__main__':
    '''
    data_loader.single_read(
        'D:\\abs_sample_gaopan\\Data\\processed_data', '1A3R_L')
    '''
    main()
