from src.data_preprocessing import load_data, clean_data, save_processed_data
from src.model_training import train_model, save_model
from src.model_evaluation import load_model
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    # Load raw data
    raw_data_path = os.path.join('data', 'raw', 'Software_engineer _salaries.csv')
    df = load_data(raw_data_path)
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Save cleaned data
    processed_data_path = os.path.join('data', 'processed', 'cleaned_software_engineer_jobs_&_salaries.csv')
    save_processed_data(df_cleaned, processed_data_path)
    
    # Load the cleaned data for training
    df_cleaned = pd.read_csv(processed_data_path)
    
    # Feature selection
    X = df_cleaned[['Company Score', 'Salary']]
    y = df_cleaned['Job Title']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Save the model
    model_save_path = os.path.join('models', 'software_engineer_salary_model.pkl')
    save_model(model, model_save_path)
    
    # Load the model for evaluation
    model = load_model(model_save_path)
    
    # Test and evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Model training and evaluation completed!")

if __name__ == "__main__":
    main()
