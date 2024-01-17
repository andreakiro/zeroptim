import matplotlib.pyplot as plt
import numpy as np


def plot_loss_and_acc_over_epochs(results):
    """Plot the loss and accuracy over training epochs"""

    epochs = range(results["n_epochs"])
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


def plot_loss_and_acc_over_iters(results):
    """Plot the loss and accuracy over training iterations"""

    iters = range(results["n_iters"])
    train_loss = results["train_loss_per_iter"]
    train_acc = results["train_acc_per_iter"]

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Primary y-axis (left)
    color = "tab:red"
    (loss_line,) = ax1.plot(iters, train_loss, color=color, label="loss")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("loss [R+]")
    ax1.set_ylim(bottom=0)

    # Second y-axis (right)
    color = "tab:blue"
    ax2 = ax1.twinx()
    (acc_line,) = ax2.plot(iters, train_acc, color=color, label="acc")
    ax2.set_ylabel("accuracy [%]")
    ax2.set_ylim(bottom=0, top=100)

    # Creating a single legend
    lines = [loss_line, acc_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    plt.show()


def plot_jpv_and_hvp_over_iters(results):
    """Plot the jvp and vhv over training iterations"""

    iters = range(results["n_iters"])
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


def remove_outliers(data):
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return data[(data >= lower_bound) & (data <= upper_bound)]


def filter_metrics_vs_loss(loss, metric):
    """Filter out outliers for a given metric"""
    loss, metric = np.array(loss), np.array(metric)
    filtered_loss = remove_outliers(loss)
    filtered_metric = remove_outliers(metric)
    mask = np.isin(loss, filtered_loss) & np.isin(metric, filtered_metric)
    return loss[mask], metric[mask]


def scatter2d(
    ax, xs, ys, title=None, xlabel=None, ylabel=None, colors=None, size=5, marker=None
):
    """Scatter plot on a matplotlib axis"""
    ax.scatter(xs, ys, s=size, c=colors, marker=marker)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def scatter2d_with_regression(
    ax,
    xs,
    ys,
    title=None,
    xlabel=None,
    ylabel=None,
    colors=None,
    size=5,
    marker=None,
    color_regression="orange",
):
    """Scatter plot with regression line on a matplotlib axis"""
    slope, intercept = np.polyfit(xs, ys, 1)
    regression = np.array(xs) * slope + intercept
    slope_str = f"slope: {round(slope, 3)}"
    ax.scatter(xs, ys, s=size, c=colors, marker=marker)
    ax.plot(xs, regression, color=color_regression, label=slope_str)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(title)


def scatter_metrics_together(metrics, bigtitle):
    """Scatter plot of metrics together"""

    loss_ = metrics["train_loss_per_iter"]
    jvps_ = metrics["jvp_per_iter"]
    vhvs_ = metrics["vhv_per_iter"]

    # Remove outliers
    loss_jvps, jvps = filter_metrics_vs_loss(loss_, jvps_)
    loss_vhvs, vhvs = filter_metrics_vs_loss(loss_, vhvs_)

    fig, axs = plt.subplots(2, 3, figsize=(26, 10))
    cmap = plt.cm.viridis

    # FULL TRAINING JVP
    colors = cmap(np.linspace(0, 1, len(jvps)))
    scatter2d_with_regression(
        axs[0, 0],
        jvps,
        loss_jvps,
        title="loss vs. jvp (full training)",
        xlabel="jvp value",
        ylabel="loss",
        colors=colors,
        size=5,
    )

    # FULL TRAINING VHV
    colors = cmap(np.linspace(0, 1, len(vhvs)))
    scatter2d_with_regression(
        axs[1, 0],
        vhvs,
        loss_vhvs,
        title="loss vs. vhv (full training)",
        xlabel="vhv value",
        ylabel="loss",
        colors=colors,
        size=5,
    )

    # EARLY TRAINING JVP
    h = len(jvps) // 2
    colors = cmap(np.linspace(0, 1, len(jvps)))[:h]
    scatter2d_with_regression(
        axs[0, 1],
        jvps[:h],
        loss_jvps[:h],
        title="loss vs. jvp (early training)",
        xlabel="jvp value",
        ylabel="loss",
        colors=colors,
        size=5,
    )

    # EARLY TRAINING VHV
    h = len(vhvs) // 2
    colors = cmap(np.linspace(0, 1, len(vhvs)))[:h]
    scatter2d_with_regression(
        axs[1, 1],
        vhvs[:h],
        loss_vhvs[:h],
        title="loss vs. vhv (early training)",
        xlabel="vhv value",
        ylabel="loss",
        colors=colors,
        size=5,
    )

    # LATE TRAINING JVP
    h = len(jvps) // 2
    colors = cmap(np.linspace(0, 1, len(jvps)))[h:]
    scatter2d_with_regression(
        axs[0, 2],
        jvps[h:],
        loss_jvps[h:],
        title="loss vs. jvp (late training)",
        xlabel="jvp value",
        ylabel="loss",
        colors=colors,
        size=5,
    )

    # LATE TRAINING VHV
    h = len(vhvs) // 2
    colors = cmap(np.linspace(0, 1, len(vhvs)))[h:]
    scatter2d_with_regression(
        axs[1, 2],
        vhvs[h:],
        loss_vhvs[h:],
        title="loss vs. vhv (late training)",
        xlabel="vhv value",
        ylabel="loss",
        colors=colors,
        size=5,
    )

    # add title to entire figure
    fig.suptitle(bigtitle, fontsize=16)
    plt.show()


def scatter_metrics_separate(results):
    """Scatter plot of metrics separately"""
    loss_ = results["train_loss_per_iter"]
    jvps_ = results["jvps_per_iter"]
    vhvs_ = results["vhvs_per_iter"]

    # Remove outliers
    loss_jvps, jvps = filter_metrics_vs_loss(loss_, jvps_)
    loss_vhvs, vhvs = filter_metrics_vs_loss(loss_, vhvs_)

    cmap = plt.cm.viridis

    # FULL TRAINING JVP

    colors = cmap(np.linspace(0, 1, len(jvps)))
    fig, ax = plt.subplots(figsize=(8, 4))
    scatter2d_with_regression(
        ax,
        jvps,
        loss_jvps,
        title="loss vs. jvp (full training)",
        xlabel="jvp value",
        ylabel="loss",
        colors=colors,
        size=5,
    )

    # FULL TRAINING VHV
    colors = cmap(np.linspace(0, 1, len(vhvs)))
    fig, ax = plt.subplots(figsize=(8, 4))
    scatter2d_with_regression(
        ax,
        vhvs,
        loss_vhvs,
        title="loss vs. vhv (full training)",
        xlabel="vhv value",
        ylabel="loss",
        colors=colors,
        size=5,
    )

    # EARLY TRAINING JVP
    h = len(jvps) // 2
    colors = cmap(np.linspace(0, 1, len(jvps)))[:h]
    fig, ax = plt.subplots(figsize=(8, 4))
    scatter2d_with_regression(
        ax,
        jvps[:h],
        loss_jvps[:h],
        title="loss vs. jvp (early training)",
        xlabel="jvp value",
        ylabel="loss",
        colors=colors,
        size=5,
    )

    # EARLY TRAINING VHV
    h = len(vhvs) // 2
    colors = cmap(np.linspace(0, 1, len(vhvs)))[:h]
    fig, ax = plt.subplots(figsize=(8, 4))
    scatter2d_with_regression(
        ax,
        vhvs[:h],
        loss_vhvs[:h],
        title="loss vs. vhv (early training)",
        xlabel="vhv value",
        ylabel="loss",
        colors=colors,
        size=5,
    )

    # LATE TRAINING JVP
    h = len(jvps) // 2
    colors = cmap(np.linspace(0, 1, len(jvps)))[h:]
    fig, ax = plt.subplots(figsize=(8, 4))
    scatter2d_with_regression(
        ax,
        jvps[h:],
        loss_jvps[h:],
        title="loss vs. jvp (late training)",
        xlabel="jvp value",
        ylabel="loss",
        colors=colors,
        size=5,
    )

    # LATE TRAINING VHV
    h = len(vhvs) // 2
    colors = cmap(np.linspace(0, 1, len(vhvs)))[h:]
    fig, ax = plt.subplots(figsize=(8, 4))
    scatter2d_with_regression(
        ax,
        vhvs[h:],
        loss_vhvs[h:],
        title="loss vs. vhv (late training)",
        xlabel="vhv value",
        ylabel="loss",
        colors=colors,
        size=5,
    )

    plt.show()
