import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


def plot_points(ax, features, targets, xlow=170, xhigh=235, ylow=2400, yhigh=6500):
    ax.scatter(features, targets, marker=".")
    ax.set_xlim(xlow, xhigh)
    ax.set_ylim(ylow, yhigh)
    ax.set_xlabel("Flipper length (mm)")
    ax.set_ylabel("Body mass (g)")


def plot_regression(ax, model, xlow=170, xhigh=235):
    model_x = np.linspace(xlow, xhigh, 1000)
    model_y = model(model_x)
    ax.plot(model_x, model_y, color="tab:orange")


def linear_regression(features, targets):
    best_fit = MLPRegressor(
        activation="relu",
        hidden_layer_sizes=(5,),
        solver="lbfgs",
        max_iter=1000,
        alpha=0,
        random_state=123,
    ).fit(features[:, np.newaxis], targets)
    return best_fit


def main():
    penguins_df = pd.read_csv("penguins.csv")
    features, targets = penguins_df.dropna()[
        ["flipper_length_mm", "body_mass_g"]
    ].values.T

    _fig, ax = plt.subplots()
    plot_points(ax, features, targets)
    best_fit = linear_regression(features, targets)
    plot_regression(ax, lambda x: best_fit.predict(x[:, np.newaxis]))
    plt.show()


if __name__ == "__main__":
    main()
