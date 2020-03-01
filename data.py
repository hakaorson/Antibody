from Preprocess import dataset
from Argset import global_args


def main():
    args = global_args.get_dataprocess_args()
    dataset.generate_dataset(args)


if __name__ == '__main__':
    main()
