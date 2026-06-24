import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Regularization_data.csv")

plt.scatter(df["Hours"], df["Marks"])

plt.xlabel("Hours")
plt.ylabel("Marks")

plt.title("Raw Dataset")

plt.show()