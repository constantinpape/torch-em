import argparse
from subprocess import run

parser = argparse.ArgumentParser()
parser.add_argument("--out", "-o", action="store_true")
args = parser.parse_args()

cmd = ["pdoc", "--docformat", "google"]

if args.out:
    cmd.extend(["--out", "tmp/"])
cmd.append("torch_em")

run(cmd)
