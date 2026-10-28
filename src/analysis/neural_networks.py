import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor


def blobs():
    blobs1 = np.concatenate(
        (
            np.random.normal(0, 1, (1000, 2)) + np.array([0, 0]),
            np.random.normal(0, 1, (1000, 2)) + np.array([3, 3]),
        )
    )
    blobs2 = np.concatenate(
        (
            np.random.normal(0, 1, (1000, 2)) + np.array([0, 3]),
            np.random.normal(0, 1, (1000, 2)) + np.array([3, 0]),
        )
    )
    shuffle = np.arange(len(blobs1) + len(blobs2))
    np.random.shuffle(shuffle)

    all_data = np.concatenate((blobs1, blobs2))[shuffle]
    targets = np.concatenate((np.zeros(len(blobs1)), np.ones(len(blobs2))))[shuffle]

    return blobs1, blobs2, all_data, targets


def fit_mlp_regressor(all_data, targets):
    best_fit = MLPRegressor(
        activation="relu", hidden_layer_sizes=(5,), max_iter=10000, alpha=0
    ).fit(all_data, targets)
    return best_fit


def plot_mlp_regressor(ax, best_fit, low=-4, high=7):
    bg_x, bg_y = np.meshgrid(np.linspace(low, high, 100), np.linspace(low, high, 100))
    bg_2d = np.column_stack((bg_x.ravel(), bg_y.ravel()))
    probs = best_fit.predict(bg_2d)
    prob_0 = probs.reshape(bg_x.shape)
    ax.contour(bg_x, bg_y, prob_0, [0.5], linestyles=["--"])
    ax.contourf(bg_x, bg_y, prob_0, alpha=0.1)


def plot_blob(ax, blob1, blob2):
    ax.scatter(blob1[:, 0], blob1[:, 1], marker=".", color="tab:blue")
    ax.scatter(blob2[:, 0], blob2[:, 1], marker=".", color="tab:orange")


def main():
    _fig, ax = plt.subplots()
    blobs1, blobs2, all_data, targets = blobs()
    best_fit = fit_mlp_regressor(all_data, targets)
    plot_blob(ax, blobs1, blobs2)
    plot_mlp_regressor(ax, best_fit)
    plt.show()


if __name__ == "__main__":
    main()
