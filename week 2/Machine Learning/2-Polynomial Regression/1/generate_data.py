import numpy as np
import pandas as pd

# Reproducible results
np.random.seed(42)

# Study hours
hours = np.arange(1, 21)

# Hidden polynomial equation
marks = 2 * hours**2 + 3 * hours

# Add some realistic noise
noise = np.random.randint(-15, 16, size=len(hours))

marks = marks + noise

# Create dataframe
df = pd.DataFrame({
    "StudyHours": hours,
    "Marks": marks
})

# Save CSV
df.to_csv("data.csv", index=False)

print("data.csv created successfully!")
print(df.head())