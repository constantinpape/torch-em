import argparse

CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]


def get_parser(default_batch_size=8, default_iterations=int(1e5)):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-p", "--phase", required=True)
    parser.add_argument("-b", "--batch_size", default=default_batch_size, type=int)
    parser.add_argument("-n", "--n_iterations", default=default_iterations, type=int)
    parser.add_argument("-s", "--save_root")
    parser.add_argument("-c", "--cell_types", nargs="+", default=CELL_TYPES)
    return parser
