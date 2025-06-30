import sys
import os
if '__file__' in globals():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
else:
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pandas as pd

def add_features(df):
    # Date parts
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Lag feature: last week's sales for the same store-dept
    df = df.sort_values(['Store', 'Dept', 'Date'])
    df['Lag_Weekly_Sales'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
    # Rolling mean of last 4 weeks
    df['Rolling_Mean_4'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(4).mean())
    
    # Encode 'Type' as numeric
    df['Type_enc'] = df['Type'].map({'A': 1, 'B': 2, 'C': 3})
    # Log transform size
    df['Log_Size'] = df['Size'].apply(lambda x: np.log1p(x))
    # Example: Is holiday week
    df['IsHoliday'] = df['IsHoliday'].astype(int)
    return df

def main():
    df = pd.read_csv('data/processed/train_cleaned.csv', parse_dates=['Date'])
    df = add_features(df)
    print(df.head())
    # Save for modeling
    df.to_csv('data/processed/train_features.csv', index=False)
    print("Saved with features to data/processed/train_features.csv")

if __name__ == '__main__':
    import numpy as np
    main()
# This script adds engineered features to the Walmart dataset.
# It includes date parts, lag features, rolling means, and encodings. 