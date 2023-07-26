import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# dark green palette
palette = sns.dark_palette("seagreen", as_cmap=True)

def load_rc_settings_general():
    sns.set_theme(style='whitegrid',rc={
        'figure.figsize':(14, 6),'font.size': 12,"axes.edgecolor": "lightgray",
        "grid.linestyle": "dashed", "grid.color": "white", "grid.color": 'lightgray',
        'patch.edgecolor': '#161616'})

def load_rc_settings_grid():
    """figuare settings for show_images_loader()"""
    sns.set_theme(
        style='whitegrid',
        rc={'figure.figsize': (12, 5), 'font.size': 10}
    )

def show_images_loader(loader, title='', feature_names=None, transform=None):
    """show 6 images from dataloader"""
    plt.close('all')

    load_rc_settings_grid()

    x, y = next(iter(loader))

    idx = np.random.randint(0, x.shape[0], 12)

    fig, ax = plt.subplots(2, 6)

    fig.suptitle(title)



    for axs, i in zip(ax.flatten(), idx):

        if transform is not None:
            x[i] = transform(x[i])

        if x.shape[1] == 3:
            axs.imshow(x[i].permute(1,2,0))
        else:
            axs.imshow(x[i].permute(1,2,0), cmap='gray')
        axs.axis(False)
        class_name = y[i].item()
        if feature_names is not None:
            class_name = feature_names[class_name]
        axs.set_title(f"{class_name}")

    plt.show()


def plot_results(
    title,
    label,
    train_results,
    test_results,
    best_epoch,
    figsize=None,
    yscale="linear",
    save_path=None,
):
    """plot loss or accuracy line after finishing training"""
    epoch_array = np.arange(len(train_results)) + 1

    sns.set(style="ticks")

    if not figsize:
        figsize = (max(5, 0.35 * (len(train_results) + 1)), 4)

    plt.figure(figsize=figsize)
    plt.plot(epoch_array, train_results, linestyle="dashed", marker="o", zorder=0)

    plt.plot(epoch_array, test_results, linestyle="dashed", marker="o", zorder=5)

    if label == 'Loss':
        plt.scatter(best_epoch + 1, min(test_results), s=100, c="red", zorder=10)
        plt.legend(["Train results", "Test results", f"Test: {min(test_results):.3f}"])
    else:
        plt.scatter(best_epoch + 1, max(test_results), s=100, c="red", zorder=10)
        plt.legend(["Train results", "Test results", f"Test: {max(test_results):.3f}"])

    plt.xlabel("Epoch")
    plt.xticks(epoch_array)

    plt.ylabel(label)
    plt.yscale(yscale)
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.title(title, y=1.1)

    sns.despine(trim=True, offset=5)
    plt.title(title, fontsize=15)

    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight")
    plt.show()


def plot_pattern(nn_binary, save_path=None):
    """plot pattern as neurons whrer red is active and black is inactive"""
    nn_num = nn_binary.shape[0]

    def drawNeuron(x, y, active=True):
        """draw circle"""
        if active:
            return plt.Circle(
                (x, y), 0.35, facecolor="red", edgecolor="#292929", linewidth=0.5
            )

        return plt.Circle(
            (x, y), 0.35, facecolor="#292929", edgecolor="#292929", linewidth=0.5
        )

    # calculate figure size
    base = int(nn_num**0.5)
    x_cor = 0
    cr = []

    for i, n in enumerate(nn_binary):
        y_cor = i % base

        if y_cor == 0:
            x_cor += 1

        cr.append(drawNeuron(x_cor, y_cor + 1, n))

    fig, ax = plt.subplots(figsize=((3 * (1 + (1 - base / x_cor))), 3), dpi=150)
    for c in cr:
        ax.add_patch(c)
    ax.set_xticks(range(0, x_cor + 2))
    ax.set_yticks(range(0, base + 2))
    ax.set_axis_off()

    if save_path:
        fig.savefig(save_path, transparent=False)

    return fig, ax


def plot_img_pattern(nn_binary, img, label, save_path=None):
    """plot pattern and an image beside it"""
    img = img.permute(1, 2, 0)
    nn_num = nn_binary.shape[0]

    def drawNeuron(x, y, active=True):
        if active:
            return plt.Circle(
                (x, y), 0.35, facecolor="red", edgecolor="#292929", linewidth=0.5
            )

        return plt.Circle(
            (x, y), 0.35, facecolor="#292929", edgecolor="#292929", linewidth=0.5
        )

    base = int(nn_num**0.5)
    x_cor = 0
    cr = []

    for i, n in enumerate(nn_binary):
        y_cor = i % base

        if y_cor == 0:
            x_cor += 1

        cr.append(drawNeuron(x_cor, y_cor + 1, n))

    fig_width = (3 * (1 + (1 - base / x_cor))) + 4
    fig, axs = plt.subplots(1, 2, figsize=(fig_width, 3), dpi=150)
    for c in cr:
        axs[1].add_patch(c)
    axs[1].set_xticks(range(0, x_cor + 2))
    axs[1].set_yticks(range(0, base + 2))
    axs[1].set_axis_off()

    # show image
    if img.shape[2] == 1:
        axs[0].imshow(img, cmap="gray")
    else:
        axs[0].imshow(img)
    axs[0].set_axis_off()

    # set title
    fig.suptitle(f"Label: {label}, active neurons: {nn_binary.sum()}")

    if save_path:
        fig.savefig(save_path, transparent=False)

    return fig, axs


def normalize_confusion_matrix(cm):
    """normalize confusion matrix number to be between 0 and 1 per row"""
    return np.divide(cm, np.tile(cm.sum(axis=0).reshape(cm.shape[0], 1), cm.shape[0]))


def save_confusion_matrix(cm, path, model_fix=None, stage="test"):
    """save confusion martix"""
    np.savetxt(path / f"{model_fix}_confusion_matrix_{stage}.txt", cm, delimiter=",")


def load_confusion_matrix(path):
    """load confusion martix"""
    return np.loadtxt(path, delimiter=",")


def plot_confusion_matrix(
    confusion_matrix,
    feature_names=None,
    map_classes=None,
    fmt=".0f",
    save_path=None,
    prefix=None,
    stage=None,
):
    """plot confusion martix"""
    ticks = range(confusion_matrix.shape[0])
    scale = max(1, (confusion_matrix.shape[0] // 10)) + 0.5
    fig, ax = plt.subplots(figsize=(9 * scale, 8 * scale))
    palette = sns.color_palette("dark:#5A9", as_cmap=True)
    # change feature names based on mapping
    # e.g {1: 1, 2: 3} where the {1: car, 3: truck}
    # benfits a model trained with skipping classes
    if map_classes:
        ticks = [*map_classes.keys()]

    if feature_names:
        if map_classes:
            ticks = [feature_names[k] for k in ticks]
        else:
            ticks = [feature_names[k] for k in ticks]

    sns.heatmap(
        confusion_matrix,
        cmap=palette,
        xticklabels=ticks,
        yticklabels=ticks,
        annot=True,
        fmt=fmt,
        ax=ax,
    )

    if prefix is not None:
        plt.title(prefix.replace("_", " ") + f" - {stage.capitalize()}")
    plt.tight_layout()

    if save_path:
        fig.savefig(
            save_path / f"{prefix}_{stage}_confusionMatrix.jpg",
            transparent=False,
            dpi=150,
        )

    return fig, ax

