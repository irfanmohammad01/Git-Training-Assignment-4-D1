# SVM Classification Example in Python
# Author: Senior ML Engineer
# Requirements: pip install scikit-learn numpy

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def run_svm_example():
    try:
        # Load the Iris dataset
        iris = datasets.load_iris()
        X = iris.data  # Features
        y = iris.target  # Labels

        # Validate dataset
        if X.size == 0 or y.size == 0:
            raise ValueError("Dataset is empty.")

        # Split into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create an SVM classifier with RBF kernel
        svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

        # Train the model
        svm_clf.fit(X_train, y_train)

        # Make predictions
        y_pred = svm_clf.predict(X_test)

        # Evaluate performance
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_svm_example()
