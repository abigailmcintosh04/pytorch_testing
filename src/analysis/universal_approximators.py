import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares


def truth(x):
    return np.where(abs(x - 0.43) < 0.15, np.sin(22 * x), -1 + 3.5 * x - 2 * x**2)


def plot_all(x, y, truey, fity):
    _fig, ax = plt.subplots()
    ax.scatter(x, y, marker=".", color="tab:orange")
    ax.plot(x, truey, color="tab:blue")
    ax.plot(x, fity, color="tab:green")
    ax.legend(["Data", "Truth", "Sigmoid"])
    plt.show()


def sigmoid(x, y, number_of_sigmoids):
    def sigmoid_component(x, center, width):
        with np.errstate(over="ignore", divide="ignore"):
            return 1 / (1 + np.exp(-(x - center) / width))

    def sigmoid_sum(x, parameters):
        out = np.zeros_like(x)
        for coefficient, center, width in parameters.reshape(-1, 3):
            out += coefficient * sigmoid_component(x, center, width)
        return out

    least_squares = LeastSquares(x, y, 0.03, sigmoid_sum)

    # Do best of 15 optimisations.
    best_minimiser = None
    for iteration in range(15):
        initial_parameters = np.zeros(number_of_sigmoids * 3)
        initial_parameters[0::3] = np.random.normal(0, 1, number_of_sigmoids)
        initial_parameters[1::3] = np.random.uniform(0, 1, number_of_sigmoids)
        initial_parameters[2::3] = np.random.normal(0, 0.1, number_of_sigmoids)

        minimiser = Minuit(least_squares, initial_parameters)
        minimiser.migrad()
        if best_minimiser is None or minimiser.fval < best_minimiser.fval:
            best_minimiser = minimiser

    y = sigmoid_sum(x, np.array(best_minimiser.values))
    return y


def main():
    x = np.linspace(0, 1, 1000)
    y = truth(x) + np.random.normal(0, 0.03, 1000)
    truey = truth(x)

    # Sigmoid fitting.
    number_of_sigmoids = 5
    fity = sigmoid(x, y, number_of_sigmoids)
    plot_all(x, y, truey, fity)


if __name__ == "__main__":
    main()
