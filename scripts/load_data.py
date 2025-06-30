import boto3
import pandas as pd
from io import StringIO

AWS_REGION = "ap-south-1"
BUCKET_NAME = "souritra-s-bucket"
PREFIX = "demand-forecasting-retail/walmart/"

files = {
    "train": f"{PREFIX}train.csv",
    "test": f"{PREFIX}test.csv",
    "features": f"{PREFIX}features.csv",
    "stores": f"{PREFIX}stores.csv"
}

s3 = boto3.client('s3', region_name=AWS_REGION)

def read_csv_from_s3(bucket, key):
    csv_obj = s3.get_object(Bucket=bucket, Key=key)
    body = csv_obj['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(body))

def main():
    df_train = read_csv_from_s3(BUCKET_NAME, files["train"])
    df_test = read_csv_from_s3(BUCKET_NAME, files["test"])
    df_features = read_csv_from_s3(BUCKET_NAME, files["features"])
    df_stores = read_csv_from_s3(BUCKET_NAME, files["stores"])
    
    print("Train Shape:", df_train.shape)
    print(df_train.head())
    print("Test Shape:", df_test.shape)
    print(df_test.head())
    print("Features Shape:", df_features.shape)
    print(df_features.head())
    print("Stores Shape:", df_stores.shape)
    print(df_stores.head())
    
    # Optionally, save locally for faster access next time
    # df_train.to_csv("../data/processed/train.csv", index=False)

if __name__ == "__main__":
    main()
