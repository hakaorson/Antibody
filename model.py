from Model import data_loader
from Argset import global_args
from Argset import args_update
from Model import trainer
from Model import base_model
from Model import model_loader
import os
import numpy as np
import torch
torch.manual_seed(1)
np.random.seed(1)


def main():
    args = global_args.get_model_args()
    all_data = data_loader.StructData(args)
    args = args_update.update(args, all_data.train_data[0])
    model = base_model.SimpleModel(args)
    for epoch in range(args.epoch):
        # print('epoch:', epoch)
        train_data_gener = data_loader.BatchGenerator(
            all_data.pos_stuct_data[:20], args.batch_size)
        # test_data = all_data.pos_stuct_data[20:]
        model = trainer.train(args, model, train_data_gener)
        if epoch % 50 == 0:
            os.makedirs(args.model_path, exist_ok=True)
            path = os.path.join(
                args.model_path, 'base_model_{0}.pt'.format(str(epoch)))
            model_loader.save_checkpoint(args, model, path)


if __name__ == '__main__':
    '''
    data_loader.single_read(
        'D:\\abs_sample_gaopan\\Data\\process_ppi', '1')
    '''
    main()
