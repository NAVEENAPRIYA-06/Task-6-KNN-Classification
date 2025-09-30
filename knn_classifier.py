import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from pandas.api.types import CategoricalDtype
DATA_FILE = 'Iris.csv' 
def plot_decision_boundary(X, y, classifier, resolution=0.02):
    """Plots the decision boundary for a 2D feature space."""
    markers = ('s', 'x', 'o', '^', 'v')
    cmap = plt.colormaps.get_cmap('Spectral') 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    categories = pd.Series(y).unique()
    cat_dtype = CategoricalDtype(categories=categories)
    Z_numeric = pd.Series(Z.ravel()).astype(cat_dtype).cat.codes.values.reshape(xx1.shape)
    y_numeric = pd.Series(y).astype(cat_dtype).cat.codes.values
    plt.contourf(xx1, xx2, Z_numeric, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl_code in enumerate(np.unique(y_numeric)):
        plt.scatter(x=X[y_numeric == cl_code, 0], y=X[y_numeric == cl_code, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=categories[idx], 
                    edgecolor='black')

if __name__ == "__main__":
    
    print("--- 1. Data Loading and Exploration ---")
    try:
        df = pd.read_csv(DATA_FILE)
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)

        print("First 5 rows of the dataset:")
        print(df.head())
        print(f"\nTotal samples: {len(df)}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {DATA_FILE}. Ensure it's in the same directory.")
        exit()

    print("\n--- 2. Feature Scaling and Data Splitting ---")
    
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} | Testing set size: {X_test.shape[0]}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features successfully scaled using StandardScaler.")
    print("\n--- 3. Basic Model Training and Evaluation (K=5) ---")
    k_value_basic = 5
    knn_classifier_basic = KNeighborsClassifier(n_neighbors=k_value_basic)
    
    # Train the model
    knn_classifier_basic.fit(X_train_scaled, y_train)
    y_pred_basic = knn_classifier_basic.predict(X_test_scaled)
    accuracy_basic = accuracy_score(y_test, y_pred_basic)
    
    print(f"Model Accuracy (K={k_value_basic}): {accuracy_basic:.4f}")

    print("\n--- 4. Finding Optimal K (Hyperparameter Tuning) ---")
    
    # Test K values from 1 up to a reasonable limit (e.g., 20)
    k_range = range(1, 21) 
    error_rate = []
    
    for i in k_range:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train_scaled, y_train)
        pred_i = knn.predict(X_test_scaled)
        error_rate.append(np.mean(pred_i != y_test))

    # Find the K with the minimum error
    optimal_k = k_range[np.argmin(error_rate)]
    min_error = min(error_rate)

    print(f"Minimum error rate: {min_error:.4f} at Optimal K = {optimal_k}")
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show() 
    print(f"\n--- 5. Final Model Training and Evaluation (Optimal K={optimal_k}) ---")
    
    # Train final model
    final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
    final_knn.fit(X_train_scaled, y_train)
    
    # Final Evaluation
    final_y_pred = final_knn.predict(X_test_scaled)
    final_accuracy = accuracy_score(y_test, final_y_pred)
    final_cm = confusion_matrix(y_test, final_y_pred)
    
    print(f"FINAL Model Accuracy (K={optimal_k}): {final_accuracy:.4f}")
    print("\nFINAL Confusion Matrix:")
    print(final_cm)

    # --- Decision Boundary Visualization ---
    print("\n--- 6. Decision Boundary Visualization (2D Plot) ---")
    print("Displaying Decision Boundary (requires closing the window to finish execution)...")
    
    # Combine train and test data for plotting
    X_combined = np.vstack((X_train_scaled, X_test_scaled))
    y_combined = np.hstack((y_train, y_test))
    
    # Select the first two features (Sepal Length, Sepal Width) for 2D visualization
    X_2d = X_combined[:, :2] 
    
    # Train a new KNN model using only the 2 selected features
    final_knn_2d = KNeighborsClassifier(n_neighbors=optimal_k)
    final_knn_2d.fit(X_2d, y_combined)

    plt.figure(figsize=(10, 7))
    plot_decision_boundary(X_2d, y_combined, final_knn_2d)
    plt.title(f'KNN Decision Boundary (K={optimal_k}) on Scaled Sepal Features (Features 1 & 2)')
    plt.xlabel('Sepal Length (Standardized)')
    plt.ylabel('Sepal Width (Standardized)')
    plt.legend(loc='upper right')
    plt.show()
    
    print("\nExecution finished.")