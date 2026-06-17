import pandas as pd

data = {
    'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
    'sales': [100, 150, 130, 170, 200]
}

df5 = pd.DataFrame(data)
print(df5)
print(df5.dtypes)

print("\nConverting 'date' column to datetime:")
df5['date'] = pd.to_datetime(df5['date'])
print(df5.dtypes)

print("\nExtracting year, month, and day from 'date' column:")
df5['year'] = df5['date'].dt.year
df5['day_name'] = df5['date'].dt.day_name()
df5['month'] = df5['date'].dt.month
print(df5)


print("\nSetting 'date' column as index:")
df5 = df5.set_index('date')
print(df5)

print("\nResampling sales data by summing every 2 days:")
resampled = df5['sales'].resample('2D').sum()
print(resampled)

print("\nCalculating rolling average of sales with a window of 2:")
df5['rolling_avg'] = df5['sales'].rolling(window=2).mean()
print(df5[['sales', 'rolling_avg']])

print("\nFiltering sales data between '2024-01-02' and '2024-01-04':")
print(df5.loc['2024-01-02':'2024-01-04'])