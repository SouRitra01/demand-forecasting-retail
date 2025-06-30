import pandas as pd
import numpy as np

def baseline_forecast():
    df = pd.read_csv('data/processed/train_features.csv', parse_dates=['Date'])
    # Naive baseline: predict this week's sales as last week's sales
    df['Pred_Baseline'] = df['Lag_Weekly_Sales']
    # Evaluate RMSE (excluding the first week, where lag is NaN)
    mask = ~df['Lag_Weekly_Sales'].isna()
    rmse = np.sqrt(((df.loc[mask, 'Weekly_Sales'] - df.loc[mask, 'Pred_Baseline']) ** 2).mean())
    print(f"Baseline RMSE: {rmse:,.2f}")
    return rmse

if __name__ == '__main__':
    baseline_forecast()
