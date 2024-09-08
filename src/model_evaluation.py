import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
import os

def load_model(model_path):
    """Load the saved model."""
    return joblib.load(model_path)

def load_test_data(file_path):
    """Load test data for evaluation."""
    return pd.read_csv(file_path)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using classification metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate classification metrics
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

if __name__ == "__main__":
    # Load model
    model_path = os.path.join('models', 'software_engineer_salary_model.pkl')
    model = load_model(model_path)
    
    # Load test data
    test_data_path = os.path.join('data', 'processed', 'cleaned_software_engineer_jobs_&_salaries.csv')
    df = load_test_data(test_data_path)
    
    # Debugging: Ensure the data is loaded correctly
    print("Loaded DataFrame head:\n", df.head())
    
    # Feature selection
    X_test = df[['Company Score', 'Salary']]
    y_test = df['Job Title']
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
