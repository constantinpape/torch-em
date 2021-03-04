import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', default='./validation_results.csv')
args = parser.parse_args()
path = args.path

df = pd.read_csv(path)
md = df.to_markdown(index=False)
print(md)
