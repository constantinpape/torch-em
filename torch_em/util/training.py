import argparse


def parser_helper(description=None, default_iterations=int(1e5)):
    description = "Run torch_em training" if description is None else description
    parser = argparse.ArgumentParser(description)
    parser.add_argument('-i', '--input', required=True,
                        help="Path to the input data, if not present an attempt will be made to download the data.")
    parser.add_argument('-n', '--n_iterations', type=int, default=default_iterations,
                        help="The number of training iterations.")
    parser.add_argument('--device', type=str, default=None)
    return parser
