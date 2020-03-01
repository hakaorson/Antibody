from Model import data_loader
from Argset import global_args
from Model import trainer
from Model import base_model
import torch
import random
torch.manual_seed(666)
random.seed(666)


def main():
    args = global_args.get_model_args()
    all_data = data_loader.StructData(args)
    # TODO args需要更新
    model = base_model.SimpleModel(args)
    for epoch in range(args.epoch):
        train_data_gener = data_loader.BatchGenerator(
            all_data.train_data, args.batch_size)
        valid_data_gener = data_loader.BatchGenerator(all_data.valid_data, -1)
        test_data_gener = data_loader.BatchGenerator(all_data.test_data, -1)
        model = trainer.train(args, model, train_data_gener)
        pass
    pass


if __name__ == '__main__':
    main()
