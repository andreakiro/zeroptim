import json
from pathlib import Path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib

matplotlib.use("macosx")  # For a native macOS backend
plt.ion()  # Turn on interactive mode

OUTPUT_DIR: Path = Path.cwd() / "outputs"
RESULT_FILE: str = "results.json"


def read(filepath):
    with open(filepath, "r") as file:
        results = json.load(file)
    return results


def plot_loss_and_acc(results):
    epochs = range(1, results["n_epochs"] + 1)
    train_loss = results["train_loss_per_epoch"]
    train_acc = results["train_acc_per_epoch"]

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Primary y-axis (left)
    color = "tab:red"
    (loss_line,) = ax1.plot(epochs, train_loss, color=color, label="loss")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss [R+]")

    # Second y-axis (right)
    color = "tab:blue"
    ax2 = ax1.twinx()
    (acc_line,) = ax2.plot(epochs, train_acc, color=color, label="acc")
    ax2.set_ylabel("accuracy [%]")

    # Creating a single legend
    lines = [loss_line, acc_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    plt.show()


def plot_jpv_and_hvp(results):
    iters = range(1, results["n_iters"] + 1)
    jvps = results["jvps_per_iter"]
    vhvs = results["vhvs_per_iter"]

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Primary y-axis (left)
    color = "tab:red"
    (loss_line,) = ax1.plot(iters, jvps, color=color, label="jvp")
    ax1.set_xlabel("optimization steps")
    ax1.tick_params(axis="y", labelcolor=color)

    # Second y-axis (right)
    color = "tab:blue"
    ax2 = ax1.twinx()
    (acc_line,) = ax2.plot(iters, vhvs, color=color, label="vhv")
    ax2.tick_params(axis="y", labelcolor=color)

    # Creating a single legend
    lines = [loss_line, acc_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    plt.show()


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
    plot_loss_and_acc(results)
    plot_jpv_and_hvp(results)
    plt.show(block=True)
