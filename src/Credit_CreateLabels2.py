# Program to create the 'delinquent' label in the application_record_clean1.csv
# based on the credit_record.csv data.

import pandas as pd
import os

# Load the data
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, '..', '..', '..', 'Datasets')
dataset_path = os.path.normpath(dataset_path)  # Clean up the path

# Read the CSV files
app_df = pd.read_csv(os.path.join(dataset_path, 'application_record_clean1.csv'))
credit_df = pd.read_csv(os.path.join(dataset_path, 'credit_record.csv'))

# Remove the entries with satus = X
credit_df = credit_df[credit_df['STATUS'] != 'X']

# Print the value counts of the STATUS column
print(credit_df['STATUS'].value_counts())

# Ensure ID columns are named the same for merging
app_id_col = 'ID'
credit_id_col = 'ID'

# Filter out application records with no matching credit records
print("Filtering application records with no matching credit records...")
matched_ids = set(credit_df[credit_id_col].unique())
app_df = app_df[app_df[app_id_col].isin(matched_ids)].copy()
print("Done filtering.  Found {} matching records.".format(len(app_df)))

# Function to determine delinquency
def is_delinquent(id_):
    statuses = credit_df.loc[credit_df[credit_id_col] == id_, 'STATUS']
    return int(any(status in ['2', '3', '4', '5', 2, 3, 4, 5] for status in statuses))

# Apply the function to each record
print("Creating delinquency labels...")
app_df['Delinquent'] = app_df[app_id_col].apply(is_delinquent)

print("Delinquency labels created.")
print("Delinquent counts:\n", app_df['Delinquent'].value_counts())

# Drop the ID column (no longer needed)
app_df.drop(columns=[app_id_col], inplace=True)

# Save the result if needed
# app_df.to_csv('credit_delinquency_v2.csv', index=False)

print(app_df.head())