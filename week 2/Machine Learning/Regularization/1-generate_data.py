import numpy as np
import pandas as pd

np.random.seed(42)

x = np.linspace(0, 10, 10)

y = 2 * x**2 + 3 * x + 5

noise = np.random.normal(0, 50, len(x))

y = y + noise

df = pd.DataFrame({
    "Hours": x,
    "Marks": y
})

df.to_csv("Regularization_data.csv", index=False)

print(df.head())
print(f"\nTotal samples: {len(df)}")