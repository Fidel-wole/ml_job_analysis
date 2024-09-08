import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def load_processed_data(file_path):
    """Load processed data for training."""
    return pd.read_csv(file_path)

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    """Save the trained model."""
    joblib.dump(model, model_path)

if __name__ == "__main__":
    # Load the cleaned data
    processed_data_path = os.path.join('data', 'processed', 'cleaned_software_engineer_jobs_&_salaries.csv')
    df = load_processed_data(processed_data_path)
    
    # Debugging: Print the head of the data to ensure it's loaded properly
    print("Loaded DataFrame head:\n", df.head())
    
    # Feature selection
    X = df[['Company Score', 'Salary']]  # Using Company Score and Salary as features
    y = df['Job Title']  # Target column
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Save the model
    model_save_path = os.path.join('models', 'software_engineer_salary_model.pkl')
    save_model(model, model_save_path)
    
    print("Model training complete and saved at:", model_save_path)
