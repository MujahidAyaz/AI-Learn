import pandas as pd
import numpy as np

data = {
    'name': ['Ali', 'Sara', 'Omar', 'Ali', None],
    'age': [25, 30, None, 25, 22],
    'city': [' Lahore', 'Karachi ', 'Multan', ' Lahore', 'Multan']
}

df2 = pd.DataFrame(data)
print(df2)

print("\n")

print(df2.info()) # This will show the number of non-null entries in each column, the data types, and memory usage.

print(df2.describe()) # This will provide summary statistics for the numeric columns in the DataFrame.


print("\n")

clean1 = df2.dropna() # This will remove any rows that contain NaN values. The resulting DataFrame will only include rows where all columns have valid (non-null) data.
print(clean1)

print("\n")

clean2 = df2.fillna({'name': 'Unknown', 'age': df2['age'].mean()}) # This will fill the NaN values in the 'name' column with 'Unknown' and the NaN values in the 'age' column with the mean age calculated from the existing age values in the DataFrame.
print(clean2)

print("\n")

print(df2.duplicated()) # This will return a boolean Series indicating whether each row is a duplicate of a previous row. The first occurrence of a duplicate will be marked as False, and subsequent occurrences will be marked as True.

print("\n")

clean3 = df2.drop_duplicates() # This will remove duplicate rows from the DataFrame. Only the first occurrence of each duplicate row will be kept, and all subsequent duplicates will be removed. The resulting DataFrame will contain only unique rows based on all columns.
print(clean3)

print("\n") 

df2['city'] = df2['city'].str.strip() # This will remove any leading and trailing whitespace characters from the strings in the 'city' column. The resulting DataFrame will have the 'city' values cleaned of any extra spaces at the beginning or end of the strings.
print(df2['city'])

print(df2['city'].str.upper()) # This will convert all the characters in the 'city' column to uppercase. The resulting Series will contain the city names in uppercase letters.

print(df2['city'].str.lower()) # This will convert all the characters in the 'city' column to lowercase. The resulting Series will contain the city names in lowercase letters.

print(df2['city'].str.title()) # This will convert the first character of each word in the 'city' column to uppercase and the remaining characters to lowercase. The resulting Series will contain the city names in title case (e.g., "Lahore" instead of "LAHORE" or "lahore").

print("\n")

df2['age'] = df2['age'].fillna(0).astype(int) # This will first fill any NaN values in the 'age' column with 0, and then convert the entire 'age' column to integers. The resulting DataFrame will have the 'age' column with all values as integers, and any missing values will be replaced with 0.
print(df2.dtypes) # This will print the data types of each column in the DataFrame. After the previous operation, the 'age' column should now have the data type 'int64', indicating that it contains integer values. The other columns will retain their original data types (e.g., 'object' for strings).
print(df2)