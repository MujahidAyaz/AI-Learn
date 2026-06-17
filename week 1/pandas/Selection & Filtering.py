import pandas as pd


data = {
    'name': ['Ali', 'Sara', 'Omar', 'Hina', 'Bilal'],
    'age': [25, 30, 22, 28, 35],
    'city': ['Lahore', 'Karachi', 'Multan', 'Lahore', 'Islamabad'],
    'marks': [85, 92, 78, 88, 95]
}

df3 = pd.DataFrame(data)
print(df3)

print("\n")
print(df3['age'] > 25) # This will return a boolean Series indicating whether each value in the 'age' column is greater than 25. The resulting Series will have True for rows where the condition is met and False for rows where it is not.

print("\n")
filtered_df = df3[df3['age'] > 25] # This will create a new DataFrame that includes only the rows where the 'age' column has values greater than 25. The resulting DataFrame will contain only the entries for Sara, Hina, and Bilal, as their ages are above 25.
print(filtered_df)

print("\n")
result = df3[(df3['age'] > 25) & (df3['marks'] > 90)] # This will create a new DataFrame that includes only the rows where both conditions are met: the 'age' column has values greater than 25 AND the 'marks' column has values greater than 90. The resulting DataFrame will contain only the entry for Bilal, as he is the only one who meets both conditions (age above 25 and marks above 90).
print(result)

print("\n") 
result2 = df3[(df3['age'] > 25) | (df3['marks'] > 90)] # This will create a new DataFrame that includes rows where either the 'age' column has values greater than 25 OR the 'marks' column has values greater than 90. The resulting DataFrame will contain the entries for Sara, Hina, and Bilal, as they either have ages above 25 or marks above 90 (Sara and Bilal have both conditions met, while Hina has only the age condition met).
print(result2)

print("\n")
result3 = df3[~(df3['age'] > 25)] # This will create a new DataFrame that includes only the rows where the 'age' column does NOT have values greater than 25. The resulting DataFrame will contain only the entries for Ali and Omar, as their ages are not above 25.
print(result3)

print("\n")
result4 = df3[df3['city'].str.contains('a')] # This will create a new DataFrame that includes only the rows where the 'city' column contains the letter 'a'. The resulting DataFrame will contain the entries for Sara, Omar, and Bilal, as their city names (Karachi, Multan, and Islamabad) contain the letter 'a'.
print(result4)

print("\n")
result5 = df3[df3['name'].str.startswith('A')] # This will create
# a new DataFrame that includes only the rows where the 'name' column starts with the letter 'A'. The resulting DataFrame will contain only the entry for Ali, as his name is the only one that starts with 'A'.
print(result5)

print("\n")
result6 = df3[df3['name'].str.endswith('a')] # This will create a new DataFrame that includes only the rows where the 'name' column ends with the letter 'a'. The resulting DataFrame will contain only the entry for Sara, as her name is the only one that ends with 'a'.
print(result6)

print("\n")
result7 = df3[df3['name'].str.len() > 4] # This will create a new DataFrame that includes only the rows where the length of the string in the 'name' column is greater than 4 characters. The resulting DataFrame will contain the entries for Sara, Omar, Hina, and Bilal, as their names have more than 4 characters (Sara has 4 characters but is included because the condition is strictly greater than 4).
print(result7)


#query method
print("\n")
result11 = df3.query('age > 25 and marks > 90') # This will create a new DataFrame that includes only the rows where the 'age' column has values greater than 25 AND the 'marks' column has values greater than 90, using the query method. The resulting DataFrame will contain only the entry for Bilal, as he is the only one who meets both conditions (age above 25 and marks above 90).
print(result11)

print("\n")
result12 = df3.query('age > 25 or marks > 90') # This will create a new DataFrame that includes rows where either the 'age' column has values greater than 25 OR the 'marks' column has values greater than 90, using the query method. The resulting DataFrame will contain the entries for Sara, Hina, and Bilal, as they either have ages above 25 or marks above 90 (Sara and Bilal have both conditions met, while Hina has only the age condition met).
print(result12)

print("\n")
result13 = df3.query('not (age > 25)') # This will create a new DataFrame that includes only the rows where the 'age' column does NOT have values greater than 25, using the query method. The resulting DataFrame will contain only the entries for Ali and Omar, as their ages are not above 25.
print(result13)

print("\n")
result14 = df3.query('city.str.contains("a")', engine='python') # This will create a new DataFrame that includes only the rows where the 'city' column contains the letter 'a', using the query method with the Python engine. The resulting DataFrame will contain the entries for Sara, Omar, and Bilal, as their city names (Karachi, Multan, and Islamabad) contain the letter 'a'.
print(result14)

print("\n")
result15 = df3.query('name.str.startswith("A")', engine='python') # This will create a new DataFrame that includes only the rows where the 'name' column starts with the letter 'A', using the query method with the Python engine. The resulting DataFrame will contain only the entry for Ali, as his name is the only one that starts with 'A'.
print(result15)

print("\n")
result16 = df3.query('name.str.endswith("a")', engine='python') #   This will create a new DataFrame that includes only the rows where the 'name' column ends with the letter 'a', using the query method with the Python engine. The resulting DataFrame will contain only the entry for Sara, as her name is the only one that ends with 'a'.
print(result16)

print("\n")
result17 = df3.query('name.str.len() > 4', engine='python') # This will create a new DataFrame that includes only the rows where the length of the string in the 'name' column is greater than 4 characters, using the query method with the Python engine. The resulting DataFrame will contain the entries for Sara, Omar, Hina, and Bilal, as their names have more than 4 characters (Sara has 4 characters but is included because the condition is strictly greater than 4).
print(result17)     

#isin method
print("\n")
result21 = df3[df3['city'].isin(['Lahore', 'Multan'])] # This will create a new DataFrame that includes only the rows where the 'city' column has values that are in the list ['Lahore', 'Multan']. The resulting DataFrame will contain the entries for Ali, Omar, and Hina, as their city names are either 'Lahore' or 'Multan'.
print(result21)

print("\n")
result22 = df3[df3['name'].isin(['Ali', 'Sara'])] # This will create a new DataFrame that includes only the rows where the 'name' column has values that are in the list ['Ali', 'Sara']. The resulting DataFrame will contain the entries for Ali and Sara, as their names are either 'Ali' or 'Sara'.
print(result22)

print("\n")
result23 = df3[df3['age'].isin([25, 30])] # This    will create a new DataFrame that includes only the rows where the 'age' column has values that are in the list [25, 30]. The resulting DataFrame will contain the entries for Ali and Sara, as their ages are either 25 or 30.
print(result23)

print("\n")
result24 = df3[df3['marks'].isin([85, 95])] # This will create a new DataFrame that includes only the rows where the 'marks' column has values that are in the list [85, 95]. The resulting DataFrame will contain the entries for Ali and Bilal, as their marks are either 85 or 95.
print(result24)

#loc method
print("\n")
result31 = df3.loc[df3['marks'] > 85, ['name', 'marks']] # This will create a new DataFrame that includes only the 'name' and 'marks' columns for the rows where the 'marks' column has values greater than 85, using the loc method. The resulting DataFrame will contain the entries for Sara, Hina, and Bilal, as their marks are above 85.
print(result31)

print("\n")
result32 = df3.loc[df3['age'] > 25, 'city'] # This will create a new Series that includes only the 'city' column for the rows where the 'age' column has values greater than 25, using the loc method. The resulting Series will contain the city names for Sara, Hina, and Bilal, as their ages are above 25.
print(result32)

print("\n")
result33 = df3.loc[df3['name'].str.contains('a'), 'age'] # This will create a new Series that includes only the 'age' column for the rows where the 'name' column contains the letter 'a', using the loc method. The resulting Series will contain the ages for Sara and Omar, as their names contain the letter 'a'.
print(result33)

