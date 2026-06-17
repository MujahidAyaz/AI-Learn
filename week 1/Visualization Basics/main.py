import pandas as pd
import matplotlib.pyplot as plt

data = {
    'day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
    'sales': [100, 150, 130, 170, 200],
    'profit': [20, 35, 25, 40, 50]
}

df6 = pd.DataFrame(data)
print(df6)

df6['sales'].plot(kind='line', color='blue', marker='o', linestyle='-', linewidth=2, markersize=8, figsize=(10, 6), grid=True)
plt.title('Daily Sales')
plt.xlabel('Day')
plt.ylabel('Sales')

plt.show()

df6.plot(x='day', y='profit', kind='bar', color='orange', edgecolor='black', alpha=0.7, figsize=(10, 6), grid=True, linewidth=1.5, linestyle='--')
plt.title('Sales by Day')
plt.show()

#df6['sales'].plot(kind='hist', bins=5)
#plt.title('Sales Distribution')
#plt.show()

import numpy as np

random_sales = pd.Series(np.random.randint(50, 200, 100))
random_sales.plot(kind='hist', bins=10 , color='green' , edgecolor='black' , alpha=0.7, figsize=(10, 6), grid=True, linewidth=1.5, linestyle='--')
plt.title('Sales Distribution (100 random values)')
plt.show()

df6.plot(x='sales', y='profit', kind='scatter', color='red')

plt.title('Sales vs Profit')
plt.show()