import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os  # Ensure os is imported

def train_model(data_path: str, model_path: str):
    # Load dataset
    data = pd.read_csv(data_path)
    X, y = data['Context'], data['NextWord']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('svm', SVC(kernel='linear', probability=True))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    # Test model
    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    # Ensure the directory for the model_path exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    joblib.dump(pipeline, model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    train_model(r'data\next_word_dataset.csv', r"model\next_word_svm.pkl")
