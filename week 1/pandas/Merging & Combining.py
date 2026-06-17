import pandas as pd

employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4],
    'name': ['Ali', 'Sara', 'Omar', 'Hina'],
    'dept_id': [10, 20, 10, 30]
})

departments = pd.DataFrame({
    'dept_id': [10, 20, 40],
    'dept_name': ['Sales', 'IT', 'HR']
})

print(employees)
print(departments)

print("\nMerging employees and departments on 'dept_id' with inner join:")
inner = pd.merge(employees, departments, on='dept_id', how='inner')
print(inner)

print("\nMerging employees and departments on 'dept_id' with left join:")
left = pd.merge(employees, departments, on='dept_id', how='left')
print(left)

print("\nMerging employees and departments on 'dept_id' with right join:")
right = pd.merge(employees, departments, on='dept_id', how='right')
print(right)

print("\nMerging employees and departments on 'dept_id' with outer join:")
outer = pd.merge(employees, departments, on='dept_id', how='outer')
print(outer)

print("\nConcatenating employees and more_employees:")
more_employees = pd.DataFrame({
    'emp_id': [5, 6],
    'name': ['Bilal', 'Zara'],
    'dept_id': [10, 20]
})

stacked = pd.concat([employees, more_employees])
print(stacked)


print("\nConcatenating employees and more_employees with ignore_index=True:")
stacked2 = pd.concat([employees, more_employees], ignore_index=True)
print(stacked2)