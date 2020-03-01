import argparse


def get_dataprocess_args():
    args = argparse.ArgumentParser()
    args.add_argument('--expand_dist', type=int, default=6)
    args.add_argument('--graph_dist', type=int, default=10)
    args.add_argument('--interface_dist', type=float, default=4.5)

    args.add_argument('--origin_path', type=str,
                      default='Data/origin_data')
    args.add_argument('--process_path', type=str,
                      default='Data/processed_data')
    result = args.parse_args()
    return result


def get_model_args():
    args = argparse.ArgumentParser()
    # data
    args.add_argument('--process_path', type=str,
                      default='Data/processed_data')
    args.add_argument('--neg_rate', type=float, default=2.0)
    args.add_argument('--split_rate', type=str, default='8 1 1')
    # train
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--epoch', type=int, default=50)
    # model
    args.add_argument('--gcn_input_size', type=int, default=7)
    args.add_argument('--gcn_hidden_size', type=int, default=32)
    args.add_argument('--gcn_output_size', type=int, default=32)
    args.add_argument('--hidden_layer', type=int, default=3)

    args.add_argument('--graph_feat_size', type=int, default=32)

    args.add_argument('--activate', type=bool, default=True)
    args.add_argument('--dropout', type=bool, default=True)
    args.add_argument('--bias', type=bool, default=True)
    result = args.parse_args()
    return result


if __name__ == '__main__':
    pass