from argparse import ArgumentParser
from pathlib import Path

import subprocess
import time
import glob

import generate as g

parser = ArgumentParser()
parser.add_argument("--sweep", type=str, help="path to config file", required=True)
parser.add_argument("--epochs", type=int, help="number of training epochs", default=5)
parser.add_argument("--iters", type=int, help="max total iterations", default=None)
args = parser.parse_args()


def run(command, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        process = subprocess.Popen(command, shell=True, stdout=f, stderr=f)
    return process.pid, out


def launch(configs):
    pids = []
    for i, config in enumerate(configs):
        timestamp = time.strftime("%Y%m%d%H%M%S")
        sweepname = args.sweep.replace("/", "-")
        output_file = f"procs/{sweepname}/{timestamp}/{i}.txt"
        command = f"python main.py --config {config} --epochs {args.epochs}"
        command = f"{command} --iters {args.iters}" if args.iters is not None else command
        pid, file = run(command, output_file)
        pids.append((pid))

    return pids


if __name__ == "__main__":
    configs = g.generate_configs_from(args.sweep)
    pids = launch(configs)
    print(pids)
