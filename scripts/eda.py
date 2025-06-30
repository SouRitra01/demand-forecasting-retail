import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from scripts.load_data import read_csv_from_s3, BUCKET_NAME, files

def eda_basic(df, name):
    print(f"--- {name} ---")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Null values:\n", df.isnull().sum())
    print("Sample data:\n", df.head())
    print("\n")

def main():
    df_train = read_csv_from_s3(BUCKET_NAME, files["train"])
    df_test = read_csv_from_s3(BUCKET_NAME, files["test"])
    df_features = read_csv_from_s3(BUCKET_NAME, files["features"])
    df_stores = read_csv_from_s3(BUCKET_NAME, files["stores"])
    
    eda_basic(df_train, "Train")
    eda_basic(df_test, "Test")
    eda_basic(df_features, "Features")
    eda_basic(df_stores, "Stores")

    print("Train duplicates:", df_train.duplicated().sum())
    print("Test duplicates:", df_test.duplicated().sum())

    print("\nTrain Summary:\n", df_train.describe())
    print("\nFeatures Summary:\n", df_features.describe())

    print("\nStore Types:\n", df_stores['Type'].value_counts())

if __name__ == "__main__":
    main()
# This script performs basic exploratory data analysis (EDA) on the datasets loaded from S3.
# It checks the shape, columns, null values, and sample data of each dataset.