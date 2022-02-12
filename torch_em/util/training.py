import argparse


def parser_helper(description=None, default_iterations=int(1e5), default_batch_size=1, require_input=True):
    description = "Run torch_em training" if description is None else description
    parser = argparse.ArgumentParser(description)
    if require_input:
        parser.add_argument("-i", "--input", required=True,
                            help="Path to the input data, if not present an attempt will be made to download the data.")
    parser.add_argument("-n", "--n_iterations", type=int, default=default_iterations,
                        help="The number of training iterations.")
    parser.add_argument("-b", "--batch_size", type=int, default=default_batch_size,
                        help="The batch size")
    parser.add_argument("--check", "-c", type=int, default=0, help="Check the data loader instead of running training.")
    parser.add_argument("--from_checkpoint", type=int, default=0, help="Start training from existing checkpoint.")
    parser.add_argument("--device", type=str, default=None)
    return parser
