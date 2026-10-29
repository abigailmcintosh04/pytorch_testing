import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

penguins_df = pd.read_csv("data/penguins.csv")


def format_tensors(features, targets):
    scaler = StandardScaler()

    features = features.to_numpy()

    features_scaled = scaler.fit_transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    targets_tensor = torch.tensor(targets.to_numpy(), dtype=torch.long)

    return features_tensor, targets_tensor, scaler


def plot_decision_boundary(features, targets, predicted_classes, model, scaler):
    species_names = ["Adelie", "Chinstrap", "Gentoo"]
    colors = ["blue", "green", "orange"]
    x_min, x_max = (
        features["bill_length_mm"].min() - 2,
        features["bill_length_mm"].max() + 2,
    )
    y_min, y_max = (
        features["bill_depth_mm"].min() - 2,
        features["bill_depth_mm"].max() + 2,
    )
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    scaled_grid = torch.tensor(scaler.transform(grid), dtype=torch.float32)

    with torch.no_grad():
        grid_predictions = torch.argmax(model(scaled_grid), dim=1).numpy()

    _fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(
        x_grid,
        y_grid,
        grid_predictions.reshape(x_grid.shape),
        levels=np.arange(len(species_names) + 1) - 0.5,
        colors=colors,
        alpha=0.2,
    )

    targets_array = targets.to_numpy()
    predicted_array = predicted_classes.numpy()
    for class_id, (species, color) in enumerate(zip(species_names, colors)):
        class_points = targets_array == class_id
        ax.scatter(
            features.loc[class_points, "bill_length_mm"],
            features.loc[class_points, "bill_depth_mm"],
            label=species,
            color=color,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.3,
        )

    incorrect = predicted_array != targets_array
    ax.scatter(
        features.loc[incorrect, "bill_length_mm"],
        features.loc[incorrect, "bill_depth_mm"],
        facecolors="none",
        edgecolors="black",
        label="Incorrect prediction",
        s=100,
    )
    ax.set_xlabel("Bill Length (mm)")
    ax.set_ylabel("Bill Depth (mm)")
    ax.set_title("Penguin classification decision boundaries")
    ax.legend()
    plt.show()


def print_accuracy(predicted_classes, targets):
    accuracy = (predicted_classes == targets).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")


def main():
    penguins = penguins_df.dropna(
        subset=["bill_length_mm", "bill_depth_mm", "species"]
    ).copy()
    features = penguins[["bill_length_mm", "bill_depth_mm"]]
    targets = penguins["species"].astype("category").cat.codes

    features_tensor, targets_tensor, scaler = format_tensors(features, targets)

    model = nn.Sequential(
        nn.Linear(2, 5),
        nn.ReLU(),
        nn.Linear(5, 3),
    )

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters())

    for _ in range(1000):
        optimiser.zero_grad()
        preds = model(features_tensor)
        loss = loss_function(preds, targets_tensor)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        logits = model(features_tensor)
        predicted_classes = torch.argmax(logits, dim=1)

    plot_decision_boundary(features, targets, predicted_classes, model, scaler)
    print_accuracy(predicted_classes, targets_tensor)


if __name__ == "__main__":
    main()
