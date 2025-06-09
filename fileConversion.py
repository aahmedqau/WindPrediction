# File conversion: extract one hour from each day and save as a new file

import pandas as pd

# Read the CSV file
data = pd.read_csv('./Data/Five_year_h_data.csv')

# Define a function to parse the date
def parse_date(date_str):
    return pd.to_datetime(date_str, format='%d/%m/%Y %H').strftime('%Y-%m-%d')

# Modify the date format
data['date'] = data['date'].apply(parse_date)

# Select one row every 24 rows (i.e., one hour per day)
filtered_data = data.iloc[12::24, :]

# Save the result to a new CSV file
filtered_data.to_csv('./Data/Five_year_d_12_data.csv', index=False)
