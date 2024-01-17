import json
from pathlib import Path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
import matplotlib
import plots
import os

matplotlib.use("macosx")  # For a native macOS backend
plt.ion()  # Turn on interactive mode

OUTPUT_DIR: Path = Path.cwd() / "outputs"
RESULT_FILE: str = "metrics.json"
FIGURES = "figures"


def read(filepath):
    with open(filepath, "r") as file:
        results = json.load(file)
    return results


def last_result_filepath():
    outputs_dir = Path(OUTPUT_DIR)
    dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()])
    if not dirs:
        raise Exception("No results found")  # noqa: E701
    return str(dirs[-1]) + "/" + RESULT_FILE


def save_figures(filename):
    if not os.path.exists(FIGURES):
        os.makedirs(FIGURES)
    figure_numbers = plt.get_fignums()
    for i, fig_num in enumerate(figure_numbers, start=1):
        fig = plt.figure(fig_num)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        filename = f"{FIGURES}/{timestamp}-{filename}-{i:02d}.png"
        fig.savefig(filename)


def parse_raw_metrics(metrics):
    parsed = {}
    m_iter = [m for m in metrics["per_iter"] if "jvp_per_iter" in m.keys()]
    parsed["train_loss_per_iter"] = list(map(lambda x: x["train_loss_per_iter"], m_iter))
    parsed["train_acc_per_iter"] = list(map(lambda x: x["train_acc_per_iter"], m_iter))
    parsed["jvp_per_iter"] = list(map(lambda x: x["jvp_per_iter"], m_iter))
    parsed["vhv_per_iter"] = list(map(lambda x: x["vhv_per_iter"], m_iter))
    return parsed


parser = ArgumentParser()
parser.add_argument("--filepath", type=str, default=None)
parser.add_argument("--bigtitle", type=str, default=None)
parser.add_argument("--save", type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    filepath = args.filepath or last_result_filepath()
    metrics = parse_raw_metrics(read(filepath))
    plots.scatter_metrics_together(metrics, args.bigtitle)
    if args.save:
        save_figures(args.save)
    plt.show(block=True)
