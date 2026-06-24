{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7d77hREEVUgD"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "\n",
        "class LinearRegression:\n",
        "\n",
        "    # runs automatically when you create the model\n",
        "    # sets up starting values\n",
        "    def __init__(self):\n",
        "        self.m = 0       # slope (theta1) — start with 0\n",
        "        self.b = 0       # intercept (theta0) — start with 0\n",
        "        self.costs = []  # empty list to track cost history\n",
        "\n",
        "    # given X, return prediction using current m and b\n",
        "    def predict(self, X):\n",
        "        return self.m * X + self.b\n",
        "\n",
        "    # training — runs gradient descent and learns m and b\n",
        "    def fit(self, X, y):\n",
        "        n = len(X)   # number of samples\n",
        "        lr = 0.01    # learning rate — how big each step is\n",
        "\n",
        "        for epoch in range(1000):   # repeat 1000 times\n",
        "\n",
        "            # Step 1: predict with current m and b\n",
        "            y_pred = self.predict(X)\n",
        "\n",
        "            # Step 2: compute cost (how wrong are we?)\n",
        "            cost = np.mean((y - y_pred) ** 2)\n",
        "            self.costs.append(cost)  # save cost for this epoch\n",
        "\n",
        "            # Step 3: compute gradients (which direction to fix m and b?)\n",
        "            dm = (-2/n) * np.sum(X * (y - y_pred))  # gradient of slope\n",
        "            db = (-2/n) * np.sum(y - y_pred)         # gradient of intercept\n",
        "\n",
        "            # Step 4: update m and b (move opposite to gradient = downhill)\n",
        "            self.m = self.m - lr * dm\n",
        "            self.b = self.b - lr * db\n",
        "\n",
        "\n",
        "# --------------------\n",
        "# RUN IT\n",
        "# --------------------\n",
        "\n",
        "X = np.array([1, 2, 3, 4, 5])\n",
        "y = np.array([30, 40, 50, 60, 70])\n",
        "\n",
        "# create model — __init__ runs here → m=0, b=0\n",
        "model = LinearRegression()\n",
        "\n",
        "# train model — gradient descent runs 1000 times\n",
        "model.fit(X, y)\n",
        "\n",
        "# see what model learned\n",
        "print(\"Slope     :\", round(model.m, 4))      # should be ≈ 10\n",
        "print(\"Intercept :\", round(model.b, 4))      # should be ≈ 20\n",
        "\n",
        "# predict new value\n",
        "print(\"Prediction for 6:\", round(model.predict(6), 4))  # should be ≈ 80"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "SThlwoWAVUjc"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mCv5ogUrVUng"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "aRQ0-SfmVUrl"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "i228H9ntVUvp"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Zeakm4NrVUzp"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "iPSkTq2BVU3n"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1EgfcSzjVU7j"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Jwn6QJMWVU_c"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
