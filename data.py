from Preprocess import dataset
from Argset import global_args


def main():
    args = global_args.get_dataprocess_args()
    # dataset.single_process(args, '1A3R_L.H_P_')
    dataset.generate_dataset(args)
    pass


if __name__ == '__main__':
    main()
