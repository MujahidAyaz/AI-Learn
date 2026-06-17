import pandas as pd

data = {
    'name': ['Ali', 'Sara', 'Omar', 'Hina', 'Bilal'],
    'age': [25, 30, 22, 28, 35],
    'city': ['Lahore', 'Karachi', 'Multan', 'Lahore', 'Islamabad'],
    'marks': [85, 92, 78, 88, 95]
}

df3 = pd.DataFrame(data)

df3['grade'] = ['A', 'A', 'B', 'A', 'A']
print(df3)


print("\nApplying transformation to marks column:")
def pass_fail(marks):
    if marks >= 80:
        return 'Pass'
    else:
        return 'Fail'

df3['result'] = df3['marks'].apply(pass_fail)
print(df3)

print("\nMapping grade points:")
grade_points = {'A': 4, 'B': 3, 'C': 2}
df3['gpa'] = df3['grade'].map(grade_points)
print(df3)

print("\nSorting by marks:")
sorted_df = df3.sort_values('marks', ascending=False)
print(sorted_df)

print("\nRenaming columns:")
df3 = df3.rename(columns={'marks': 'score', 'city': 'location'})
print(df3.columns)

print("\nReplacing location names:")
df3['location'] = df3['location'].replace({'Lahore': 'LHR', 'Karachi': 'KHI'})
print(df3['location'])