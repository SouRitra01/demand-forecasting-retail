# scripts/data_loader.py
import sys
import os

def configure_project_path():
    if '__file__' in globals():
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    else:
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

configure_project_path()
from scripts.load_data import read_csv_from_s3, BUCKET_NAME, files

def get_all_walmart_data():
    df_train = read_csv_from_s3(BUCKET_NAME, files["train"])
    df_test = read_csv_from_s3(BUCKET_NAME, files["test"])
    df_features = read_csv_from_s3(BUCKET_NAME, files["features"])
    df_stores = read_csv_from_s3(BUCKET_NAME, files["stores"])
    return df_train, df_test, df_features, df_stores
