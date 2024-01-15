import json
from pathlib import Path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib
import plots

matplotlib.use("macosx")  # For a native macOS backend
plt.ion()  # Turn on interactive mode

OUTPUT_DIR: Path = Path.cwd() / "outputs"
RESULT_FILE: str = "results.json"


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


parser = ArgumentParser()
parser.add_argument("--filepath", type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    filepath = args.filepath or last_result_filepath()
    results = read(filepath)
    plots.scatter_metrics_separate(results)
    plt.show(block=True)
