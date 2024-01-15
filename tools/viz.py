import json
from pathlib import Path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib
import numpy as np

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


def plot_scatter_plots_v1(results):
    loss = results["train_loss_per_iter"]
    jvps = results["jvps_per_iter"]
    vhvs = results["vhvs_per_iter"]
    half = len(loss) // 2

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Full scatter plot
    axs[0].scatter(loss, jvps, c="red", label="jvp")
    axs[0].scatter(loss, vhvs, c="blue", label="vhv")
    axs[0].set_xlabel("loss")
    axs[0].set_title("jvp/vhv vs. loss (full)")
    axs[0].legend()

    # Early training scatter plot
    axs[1].scatter(loss[:half], jvps[:half], c="red", label="jvp")
    axs[1].scatter(loss[:half], vhvs[:half], c="blue", label="vhv")
    axs[1].set_xlabel("loss")
    axs[1].set_title("jvp/vhv vs. loss (early training)")
    axs[1].legend()

    # Late training scatter plot
    axs[2].scatter(loss[half:], jvps[half:], c="red", label="jvp")
    axs[2].scatter(loss[half:], vhvs[half:], c="blue", label="vhv")
    axs[2].set_xlabel("loss")
    axs[2].set_title("jvp/vhv vs. loss (late training)")
    axs[2].legend()

    plt.show()


def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return [d for d in data if lower_bound <= d <= upper_bound]


def plot_scatter_plots_v2_separate(results):
    loss_ = results["train_loss_per_iter"]
    jvps_ = results["jvps_per_iter"]
    vhvs_ = results["vhvs_per_iter"]

    # Remove outliers
    filtered_loss = remove_outliers(loss_)
    filtered_jvps = remove_outliers(jvps_)
    filtered_vhvs = remove_outliers(vhvs_)

    # filter out outliers
    filtered_pairs_jvps = [
        (l, j)
        for l, j in zip(loss_, jvps_)
        if l in filtered_loss and j in filtered_jvps
    ]
    filtered_pairs_vhvs = [
        (l, v)
        for l, v in zip(loss_, vhvs_)
        if l in filtered_loss and v in filtered_vhvs
    ]
    loss_jvps, jvps = zip(*filtered_pairs_jvps)
    loss_vhvs, vhvs = zip(*filtered_pairs_vhvs)

    # FULL TRAINING JVP
    slope, intercept = np.polyfit(jvps, loss_jvps, 1)
    regression_line = np.array(jvps) * slope + intercept
    plt.figure(figsize=(8, 6))
    plt.scatter(jvps, loss_jvps, s=5)
    plt.plot(jvps, regression_line, color="orange")
    plt.xlabel("jvp value")
    plt.ylabel("loss")
    plt.title("loss vs. jvp (full training)")
    plt.show()

    # FULL TRAINING VHV
    slope, intercept = np.polyfit(vhvs, loss_vhvs, 1)
    regression_line = np.array(vhvs) * slope + intercept
    plt.figure(figsize=(8, 6))
    plt.scatter(vhvs, loss_vhvs, s=5)
    plt.plot(vhvs, regression_line, color="orange")
    plt.xlabel("vhv value")
    plt.ylabel("loss")
    plt.title("loss vs. vhv (full training)")
    plt.show()

    # EARLY TRAINING JVP
    h = len(jvps) // 2
    slope, intercept = np.polyfit(jvps[:h], loss_jvps[:h], 1)
    regression_line = np.array(jvps[:h]) * slope + intercept
    plt.figure(figsize=(8, 6))
    plt.scatter(jvps[:h], loss_jvps[:h], s=5)
    plt.plot(jvps[:h], regression_line, color="orange")
    plt.xlabel("jvp value")
    plt.ylabel("loss")
    plt.title("loss vs. jvp (early training)")
    plt.show()

    # EARLY TRAINING VHV
    h = len(vhvs) // 2
    slope, intercept = np.polyfit(vhvs[:h], loss_vhvs[:h], 1)
    regression_line = np.array(vhvs[:h]) * slope + intercept
    plt.figure(figsize=(8, 6))
    plt.scatter(vhvs[:h], loss_vhvs[:h], s=5)
    plt.plot(vhvs[:h], regression_line, color="orange")
    plt.xlabel("vhv value")
    plt.ylabel("loss")
    plt.title("loss vs. vhv (early training)")
    plt.show()

    # LATE TRAINING JVP
    h = len(jvps) // 2
    slope, intercept = np.polyfit(jvps[h:], loss_jvps[h:], 1)
    regression_line = np.array(jvps[h:]) * slope + intercept
    plt.figure(figsize=(8, 6))
    plt.scatter(jvps[h:], loss_jvps[h:], s=5)
    plt.plot(jvps[h:], regression_line, color="orange")
    plt.xlabel("jvp value")
    plt.ylabel("loss")
    plt.title("loss vs. jvp (late training)")
    plt.show()

    # LATE TRAINING VHV
    h = len(vhvs) // 2
    slope, intercept = np.polyfit(vhvs[h:], loss_vhvs[h:], 1)
    regression_line = np.array(vhvs[h:]) * slope + intercept
    plt.figure(figsize=(8, 6))
    plt.scatter(vhvs[h:], loss_vhvs[h:], s=5)
    plt.plot(vhvs[h:], regression_line, color="orange")
    plt.xlabel("vhv value")
    plt.ylabel("loss")
    plt.title("loss vs. vhv (late training)")
    plt.show()


def plot_scatter_plots_v2_together(results):
    loss_ = results["train_loss_per_iter"]
    jvps_ = results["jvps_per_iter"]
    vhvs_ = results["vhvs_per_iter"]

    fig, axs = plt.subplots(2, 3, figsize=(24, 8))

    # Remove outliers
    filtered_loss = remove_outliers(loss_)
    filtered_jvps = remove_outliers(jvps_)
    filtered_vhvs = remove_outliers(vhvs_)

    # filter out outliers
    filtered_pairs_jvps = [
        (l, j)
        for l, j in zip(loss_, jvps_)
        if l in filtered_loss and j in filtered_jvps
    ]
    filtered_pairs_vhvs = [
        (l, v)
        for l, v in zip(loss_, vhvs_)
        if l in filtered_loss and v in filtered_vhvs
    ]
    loss_jvps, jvps = zip(*filtered_pairs_jvps)
    loss_vhvs, vhvs = zip(*filtered_pairs_vhvs)

    # FULL TRAINING JVP
    slope, intercept = np.polyfit(jvps, loss_jvps, 1)
    regression_line = np.array(jvps) * slope + intercept
    slope_str = f"slope: {round(slope, 0)}"
    axs[0, 0].scatter(jvps, loss_jvps, s=5)
    axs[0, 0].plot(jvps, regression_line, color="orange", label=slope_str)
    axs[0, 0].set_xlabel("jvp value")
    axs[0, 0].set_ylabel("loss")
    axs[0, 0].legend()
    axs[0, 0].set_title("loss vs. jvp (full training)")

    # FULL TRAINING VHV
    slope, intercept = np.polyfit(vhvs, loss_vhvs, 1)
    regression_line = np.array(vhvs) * slope + intercept
    slope_str = f"slope: {round(slope, 0)}"
    axs[1, 0].scatter(vhvs, loss_vhvs, s=5)
    axs[1, 0].plot(vhvs, regression_line, color="orange", label=slope_str)
    axs[1, 0].set_xlabel("vhv value")
    axs[1, 0].set_ylabel("loss")
    axs[1, 0].legend()
    axs[1, 0].set_title("loss vs. vhv (full training)")

    # EARLY TRAINING JVP
    h = len(jvps) // 2
    slope, intercept = np.polyfit(jvps[:h], loss_jvps[:h], 1)
    regression_line = np.array(jvps[:h]) * slope + intercept
    slope_str = f"slope: {round(slope, 0)}"
    axs[0, 1].scatter(jvps[:h], loss_jvps[:h], s=5)
    axs[0, 1].plot(jvps[:h], regression_line, color="orange", label=slope_str)
    axs[0, 1].set_xlabel("jvp value")
    axs[0, 1].set_ylabel("loss")
    axs[0, 1].legend()
    axs[0, 1].set_title("loss vs. jvp (early training)")

    # EARLY TRAINING VHV
    h = len(vhvs) // 2
    slope, intercept = np.polyfit(vhvs[:h], loss_vhvs[:h], 1)
    regression_line = np.array(vhvs[:h]) * slope + intercept
    slope_str = f"slope: {round(slope, 0)}"
    axs[1, 1].scatter(vhvs[:h], loss_vhvs[:h], s=5)
    axs[1, 1].plot(vhvs[:h], regression_line, color="orange", label=slope_str)
    axs[1, 1].set_xlabel("vhv value")
    axs[1, 1].set_ylabel("loss")
    axs[1, 1].legend()
    axs[1, 1].set_title("loss vs. vhv (early training)")

    # LATE TRAINING JVP
    h = len(jvps) // 2
    slope, intercept = np.polyfit(jvps[h:], loss_jvps[h:], 1)
    regression_line = np.array(jvps[h:]) * slope + intercept
    slope_str = f"slope: {round(slope, 0)}"
    axs[0, 2].scatter(jvps[h:], loss_jvps[h:], s=5)
    axs[0, 2].plot(jvps[h:], regression_line, color="orange", label=slope_str)
    axs[0, 2].set_xlabel("jvp value")
    axs[0, 2].set_ylabel("loss")
    axs[0, 2].legend()
    axs[0, 2].set_title("loss vs. jvp (late training)")

    # LATE TRAINING VHV
    h = len(vhvs) // 2
    slope, intercept = np.polyfit(vhvs[h:], loss_vhvs[h:], 1)
    regression_line = np.array(vhvs[h:]) * slope + intercept
    slope_str = f"slope: {round(slope, 0)}"
    axs[1, 2].scatter(vhvs[h:], loss_vhvs[h:], s=5)
    axs[1, 2].plot(vhvs[h:], regression_line, color="orange", label=slope_str)
    axs[1, 2].set_xlabel("vhv value")
    axs[1, 2].set_ylabel("loss")
    axs[1, 2].legend()
    axs[1, 2].set_title("loss vs. vhv (late training)")

    plt.legend()
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
    # plot_scatter_plots(results)
    plot_scatter_plots_v2_together(results)
    plt.show(block=True)
