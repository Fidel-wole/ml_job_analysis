import pandas as pd
import os
import re

def load_data(file_path):
    """Load the CSV file."""
    return pd.read_csv(file_path)

def clean_salary_column(salary_str):
    """Clean and convert salary strings to numeric values."""
    if pd.isna(salary_str):
        return None
    
    # Remove currency symbols and spaces
    salary_str = re.sub(r'[\$,]', '', salary_str.strip())
    
    # Handle 'K' to convert to thousands (e.g., '68K' -> 68000)
    if 'K' in salary_str:
        salary_str = salary_str.replace('K', '000')
    
    # Handle salary ranges by taking the average
    if '-' in salary_str:
        salary_range = salary_str.split('-')
        salary_low = salary_range[0].strip()
        salary_high = salary_range[1].split()[0].strip()  # Remove any extra text
        try:
            salary_avg = (float(salary_low) + float(salary_high)) / 2
            return salary_avg
        except ValueError:
            return None
    else:
        # Convert single salary values directly
        try:
            salary_num = re.sub(r'[^\d.]', '', salary_str)  # Remove any remaining non-numeric characters
            if salary_num:
                return float(salary_num)
        except ValueError:
            return None
    
    return None

def clean_data(df):
    """Handle missing values, outliers, and data types."""
    # Drop rows with missing values in the Salary column
    df = df[df['Salary'].notna()].copy() 
    
    # Clean the Salary column
    df.loc[:, 'Salary'] = df['Salary'].apply(clean_salary_column)
    
    # Drop rows where Salary could not be converted
    df = df[df['Salary'].notna()]
    
    # After cleaning, print how many rows are left
    print(f"Rows remaining after cleaning: {df.shape[0]}")
    
    # Print column names to debug
    print("Columns in DataFrame:", df.columns)

    return df

def save_processed_data(df, save_path):
    """Save the processed data to the processed folder."""
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    # Load raw data
    raw_data_path = os.path.join('data', 'raw', 'Software_engineer _salaries.csv')
    df = load_data(raw_data_path)
    
    # Print the first few rows and columns of the DataFrame
    print("DataFrame head:\n", df.head())
    print("Columns in DataFrame:", df.columns)
    
    # Clean the data
    df_cleaned = clean_data(df)
    
    # Save the cleaned data
    processed_data_path = os.path.join('data', 'processed', 'cleaned_software_engineer_jobs_&_salaries.csv')
    save_processed_data(df_cleaned, processed_data_path)

    print(f"Cleaned data saved to {processed_data_path}")
