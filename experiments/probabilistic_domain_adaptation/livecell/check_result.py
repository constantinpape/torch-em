import argparse
import pandas as pd


def check_result(path):
    table = pd.read_csv(path)
    print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        check_result(path)


if __name__ == "__main__":
    main()
