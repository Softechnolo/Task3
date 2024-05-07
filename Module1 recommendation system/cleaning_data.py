import pandas as pd
import numpy as np
from Levenshtein import ratio
import pycountry

# Load data
df = pd.read_csv('data.csv', encoding='latin1')  # Adjust encoding as per your specific dataset

# Function to remove special characters but keep whitespaces
def remove_special_chars(val):
    if isinstance(val, str):
        return ''.join(e for e in val if e.isalnum() or e.isspace())
    else:
        return val

# Apply the function to all columns
df = df.applymap(remove_special_chars)

# Handle missing values by replacing them with NaN
df.replace('', np.nan, inplace=True)

# Correcting data types
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Clean text columns using regular expressions
cols_to_clean = ['StockCode', 'Description', 'Country']
for col in cols_to_clean:
    df[col] = df[col].str.replace('[^a-zA-Z0-9 \n\.]', '')

# Handle specific issues like replacing 'Ww' with '' in UnitPrice and converting it to numeric type
df['UnitPrice'] = df['UnitPrice'].str.replace('Ww', '')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')

# Function to find closest country name
def closest_country(input_country):
    countries = [country.name for country in pycountry.countries]
    similarity_ratios = [ratio(input_country, country) for country in countries]
    max_ratio_index = np.argmax(similarity_ratios)
    return countries[max_ratio_index]

# Apply the function to the 'Country' column
df['Country'] = df['Country'].apply(closest_country)

# Remove non-numeric characters from 'InvoiceNo', 'StockCode', and 'CustomerID'
df['InvoiceNo'] = df['InvoiceNo'].str.replace('[^0-9]', '')
df['StockCode'] = df['StockCode'].str.replace('[^0-9]', '')
df['CustomerID'] = df['CustomerID'].str.replace('[^0-9]', '')

# Convert 'CustomerID' and 'StockCode' to numeric
df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
df['StockCode'] = pd.to_numeric(df['StockCode'], errors='coerce')

# Now you can fill NaN values with max value + 1
df['CustomerID'].fillna(df['CustomerID'].max() + 1, inplace=True)
df['StockCode'].fillna(df['StockCode'].max() + 1, inplace=True)

# Fill missing values in the 'Description' column with an empty string
df['Description'].fillna('', inplace=True)

# Remove duplicates by InvoiceNo
df = df.drop_duplicates(subset='InvoiceNo', keep='first')

# Remove duplicates by 'Description'
df = df.drop_duplicates(subset='Description', keep='first')

# Save the cleaned dataframe to a new CSV file
df.to_csv('cleaned_data.csv', index=False)