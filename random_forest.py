# Random Forest Classifier Example in Python
# Author: Senior Programming Expert

import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    try:
        # Load the Iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # Split into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create the Random Forest Classifier
        rf_model = RandomForestClassifier(
            n_estimators=100,      # Number of trees
            max_depth=None,        # No max depth limit
            random_state=42,       # Reproducibility
            n_jobs=-1              # Use all CPU cores
        )

        # Train the model
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Evaluate the model
        print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
        print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
        print("\n🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    except ImportError as e:
        print("❌ Missing library:", e)
        print("Install required packages using: pip install scikit-learn numpy")
    except Exception as e:
        print("❌ An error occurred:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
