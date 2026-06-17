#structured pandas learning path:
#Step 1: Basics
import pandas as pd
#pandas provides two types of classes for handling data:
#1: Series.
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)

#2: DataFrame
data = {
    'name': ['Ali', 'Sara', 'Omar'],
    'age': [25, 30, 22]
}

df = pd.DataFrame(data)
print(df)

print("\n")

print(df['name']) # Accessing the 'name' column of the DataFrame
print(df['age']) # Accessing the 'age' column of the DataFrame

print("\n")

#.loc → use names/labels (like 'age', 'name')
#.iloc → use numbers/positions (like 0, 1, 2)

print (df.loc[0])   # Accessing the first row using loc
print(df.loc[1, 'age']) # Accessing the age of the second row using loc

print("\n")

print(df.iloc[1, 1]) # Accessing the age of the second row using iloc

print("\n")

print(df.shape) # Printing the shape of the DataFrame
print(df.dtypes) # Printing the data types of each column
print(df.head(2)) # Printing the first 2 rows of the DataFrame

print("\n")


#Step 2: Data I/O


df.to_csv('students.csv', index=False) # Saving the DataFrame to a CSV file without the index
new_df = pd.read_csv('students.csv') # Reading the CSV file back into a new DataFrame
print(new_df) # Printing the new DataFrame read from the CSV file


print("\n")

df.to_excel('students.xlsx', index=False) # Saving the DataFrame to an Excel file without the index
new_df2 = pd.read_excel('students.xlsx') # Reading the Excel file back into a new DataFrame
print(new_df2) # Printing the new DataFrame read from the Excel file

print("\n")

df.to_json('students.json', orient='records') # Saving the DataFrame to a JSON file with records orientation
new_df3 = pd.read_json('students.json', orient='records') # Reading the JSON file back into a new DataFrame
print(new_df3)

print("\n")

preview = pd.read_csv('students.csv', nrows=2) # Reading only the first 2 rows of the CSV file
print(preview)