import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim


def format_tensors(features, targets):
    scaler = StandardScaler()

    features = features.to_numpy()
    targets = targets.to_numpy()

    features_scaled = scaler.fit_transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    targets_tensor = torch.tensor(targets.reshape(-1, 1), dtype=torch.float32)

    return features_tensor, targets_tensor


def plot_regression(y_pred, y_true):
    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    min_val = min(y_pred.min(), y_true.min())
    max_val = max(y_pred.max(), y_true.max())
    xlow, xhigh = min_val - 5, max_val + 5
    ylow, yhigh = min_val - 5, max_val + 5
    ax1.scatter(y_pred, y_true, marker=".")
    ax1.plot([xlow, xhigh], [ylow, yhigh], color="red", linestyle="--")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, marker=".")
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residuals")
    plt.show()


def print_metrics(y_pred, y_true):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")


def main():
    housing_df = pd.read_csv(
        "data/boston-house-prices.csv",
        sep=r"\s+",
        header=None,
        names=[
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
            "MEDV",
        ],
    )

    features = housing_df.drop(columns="MEDV")
    targets = housing_df["MEDV"]

    features_tensor, targets_tensor = format_tensors(features, targets)

    model = nn.Sequential(
        nn.Linear(13, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )

    loss_function = nn.MSELoss()
    optimiser = optim.Rprop(model.parameters())

    for _ in range(1000):
        optimiser.zero_grad()
        preds = model(features_tensor)
        loss = loss_function(preds, targets_tensor)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        y_pred = model(features_tensor).flatten().numpy()
        y_true = targets_tensor.flatten().numpy()

    plot_regression(y_pred, y_true)
    print_metrics(y_pred, y_true)


if __name__ == "__main__":
    main()
