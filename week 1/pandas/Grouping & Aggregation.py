import pandas as pd

data = {
    'name': ['Ali', 'Sara', 'Omar', 'Hina', 'Bilal', 'Zara'],
    'city': ['Lahore', 'Karachi', 'Lahore', 'Multan', 'Karachi', 'Multan'],
    'department': ['Sales', 'Sales', 'IT', 'IT', 'Sales', 'IT'],
    'salary': [50000, 60000, 55000, 45000, 62000, 48000]
}

df4 = pd.DataFrame(data)
print(df4)

print("\nGrouping by city and calculating mean salary:")
print(df4.groupby('city')['salary'].mean())

print("\nGrouping by department and calculating multiple aggregations:")
result = df4.groupby('department')['salary'].agg(['mean', 'min', 'max', 'count'])
print(result)

print("\nGrouping by city and department, then calculating mean salary:")
result2 = df4.groupby(['city', 'department'])['salary'].mean()
print(result2)

print("\nCreating a pivot table to summarize mean salary by city and department:")
pivot = df4.pivot_table(values='salary', index='city', columns='department', aggfunc='mean')
print(pivot)

print("\nCreating a crosstab to show the count of employees in each city and department:")
cross = pd.crosstab(df4['city'], df4['department'])
print(cross)