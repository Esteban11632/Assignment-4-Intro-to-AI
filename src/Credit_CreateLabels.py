# Program to create the 'delinquent' label in the application_record_clean1.csv
# based on the credit_record.csv data.

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
print(f"Value counts of the STATUS column: {credit_df['STATUS'].value_counts()}")

# Print the total observations in the credit_df
print(f"Total observations in the credit_df: {credit_df[credit_df['STATUS'].isin(['C', '0', '1'])]['ID'].value_counts()}")

# Ensure ID columns are named the same for merging
app_id_col = 'ID'
credit_id_col = 'ID'

# Filter out application records with no matching credit records
print("Filtering application records with no matching credit records...")
matched_ids = set(credit_df[credit_id_col].unique())
app_df = app_df[app_df[app_id_col].isin(matched_ids)].copy()
print("Done filtering.  Found {} matching records.".format(len(app_df)))

# Group by ID and calculate fractions for each status
status_fractions = credit_df[credit_df['STATUS'].isin(['C', '0', '1'])].groupby(credit_id_col)['STATUS'].value_counts(normalize=True).unstack(fill_value=0)

# Rename columns to indicate they are fractions
status_fractions.columns = [f'{col}_fraction' for col in status_fractions.columns]

# Add these new columns to app_df
app_df = app_df.join(status_fractions, on=app_id_col)

# Function to determine delinquency
def is_delinquent(customer_id):
    # Define what counts as a delinquent status
    delinquent_statuses = ['2', '3', '4', '5', 2, 3, 4, 5]
    
    # Get all status records for this customer
    customer_records = credit_df[credit_df[credit_id_col] == customer_id]
    customer_statuses = customer_records['STATUS']
    
    # Check if customer has any delinquent status
    has_delinquent_status = any(status in delinquent_statuses for status in customer_statuses)
    
    # Return 1 if delinquent, 0 if not
    return int(has_delinquent_status)

# Apply the function to each record
print("Creating delinquency labels...")
app_df['Delinquent'] = app_df[app_id_col].apply(is_delinquent)

print("Delinquency labels created.")
print("Delinquent counts:\n", app_df['Delinquent'].value_counts())

# Drop the ID column (no longer needed)
app_df.drop(columns=[app_id_col], inplace=True)

# Save the result if needed
app_df.to_csv(os.path.join(dataset_path, 'credit_delinquency_v2.csv'), index=False)

print(app_df.head())