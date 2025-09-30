import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
def load_and_explore_data(file_path):
    """Loads the dataset and displays basic information."""
    print("--- Loading Data ---")
    try:
        df = pd.read_csv("Iris.csv")

        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)

        print("First 5 rows of the dataset:")
        print(df.head())
        print("\nDataset info:")
        df.info()

        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
DATA_FILE = 'Iris.csv' 

iris_df = load_and_explore_data(DATA_FILE)

if iris_df is not None:
    print("\nData loaded successfully.")