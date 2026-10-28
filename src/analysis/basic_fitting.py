import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from sklearn.linear_model import LinearRegression


def manual_linear_regression(x, y):
    # Minimise chi-squared.
    sum1 = len(x)
    sumx = np.sum(x)
    sumy = np.sum(y)
    sumxx = np.sum(x**2)
    sumxy = np.sum(x * y)
    delta = sum1 * sumxx - sumx**2

    a = (sum1 * sumxy - sumx * sumy) / delta
    b = (sumxx * sumy - sumx * sumxy) / delta

    return a, b


def plot_linear_fit(x, y, a, b):
    _fig, ax = plt.subplots()
    ax.scatter(x, y, marker="+", color="tab:orange")
    ax.plot(x, a * x + b, color="tab:blue")
    plt.show()


def linear_regression(x, y):
    X = x.reshape(len(x), 1)
    best_fit = LinearRegression().fit(X, y)
    a = best_fit.coef_[0]
    b = best_fit.intercept_
    return a, b


def non_linear_function(t):
    def y(t):
        y0, t0, mu, tf = 60, 3, 0.05, 2
        return y0 - (1 / mu) * np.log(np.cosh(t - t0) / tf)

    def measurement_error(n):
        return np.random.normal(0, 1, n)

    y = y(t) + measurement_error(len(t))

    return y


def plot_non_linear(t, y, y_from_fit):
    _fig, ax = plt.subplots()
    ax.set_xlabel("t")
    ax.set_ylabel("y")

    ax.plot(t, y_from_fit, color="tab:blue")
    ax.scatter(t, y, marker=".", color="tab:orange")
    ax.errorbar(t, y, yerr=1, fmt="none", color="tab:orange")

    plt.show()


def non_linear_regression(t, y):
    def ansatz(t, y0, t0, mu, tf):
        return y0 - (1 / mu) * np.log(np.cosh(t - t0) / tf)

    least_squares = LeastSquares(t, y, 1, ansatz)
    minimiser = Minuit(least_squares, y0=100, t0=0, mu=0.1, tf=3)  # Initial guess.
    minimiser.migrad()  # Perform the minimisation.

    y_from_fit = ansatz(t, **{p.name: p.value for p in minimiser.params})

    return y_from_fit


def do_linear_regression():
    # Uniformly random x values.
    x = np.random.uniform(-5, 5, 100)

    # Linear relationship with noise.
    y = (x * 2 + 3) + np.random.normal(0, 3, 100)

    # Linear Regression.
    a, b = manual_linear_regression(x, y)
    a, b = linear_regression(x, y)
    plot_linear_fit(x, y, a, b)
    print(a, b)


def do_non_linear_regression():
    t = np.linspace(0, 10, 50)
    y = non_linear_function(t)
    y_from_fit = non_linear_regression(t, y)
    plot_non_linear(t, y, y_from_fit)


def main():

    # Linear Regression.
    # do_linear_regression()

    # Non-linear regression.
    do_non_linear_regression()


if __name__ == "__main__":
    main()
