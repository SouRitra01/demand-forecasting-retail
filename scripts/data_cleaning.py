import sys
import os

# Add project root to sys.path for import
if '__file__' in globals():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
else:
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pandas as pd
from scripts.load_data import read_csv_from_s3, BUCKET_NAME, files

def clean_and_merge():
    # Load data
    df_train = read_csv_from_s3(BUCKET_NAME, files["train"])
    df_features = read_csv_from_s3(BUCKET_NAME, files["features"])
    df_stores = read_csv_from_s3(BUCKET_NAME, files["stores"])

    # Convert dates
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_features['Date'] = pd.to_datetime(df_features['Date'])

    # Merge features and stores into train
    df = pd.merge(df_train, df_features, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = pd.merge(df, df_stores, on='Store', how='left')

    # Handle missing values: fillna or drop
    df = df.sort_values(['Store', 'Dept', 'Date'])
    # Forward-fill markdowns for each store/department (if appropriate)
    for col in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
        if col in df.columns:
            df[col] = df.groupby(['Store', 'Dept'])[col].ffill().bfill()
            df[col] = df[col].fillna(0)
    # Fill CPI, Unemployment with forward fill, then median
    for col in ['CPI', 'Unemployment']:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
            df[col] = df[col].fillna(df[col].median())

    # Reset index
    df = df.reset_index(drop=True)
    return df

def main():
    df = clean_and_merge()
    print("After merging and cleaning:")
    print(df.info())
    print(df.isnull().sum())
    # Save to processed data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/train_cleaned.csv', index=False)
    print("Saved cleaned train data to data/processed/train_cleaned.csv")

if __name__ == '__main__':
    main()
# This script loads the Walmart datasets, merges them, cleans the data, and saves the cleaned dataset.
# It handles missing values by forward-filling markdowns and filling CPI/Unemployment with median values.
# The cleaned data is saved to a specified directory for further analysis or modeling.