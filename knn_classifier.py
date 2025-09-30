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

if iris_df is not None:
    # Separate features (X) and target (y)
    # Assuming the last column is the target (Species)
    X = iris_df.iloc[:, :-1].values  
    y = iris_df.iloc[:, -1].values

    print("\n--- Feature Scaling and Data Splitting ---")

    # 1. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # 2. Feature Scaling
    # Standardize the features (important for distance-based algorithms like KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Features successfully scaled (Standardization).")
    global X_train_s, X_test_s, y_t, y_v
    X_train_s, X_test_s, y_t, y_v = X_train_scaled, X_test_scaled, y_train, y_test

if 'X_train_s' in globals(): 

    print("\n--- Training and Evaluation (K=5) ---")

    k_value = 5
    knn_classifier = KNeighborsClassifier(n_neighbors=k_value)

    # Train the model
    knn_classifier.fit(X_train_s, y_t)

    # Make predictions on the test set
    y_pred = knn_classifier.predict(X_test_s)

    # Evaluate performance
    accuracy = accuracy_score(y_v, y_pred)
    cm = confusion_matrix(y_v, y_pred)

    print(f"Model Accuracy (K={k_value}): {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
