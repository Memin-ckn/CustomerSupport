import pandas as pd

# Load the dataset
file_path = 'training_turkish_2.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
def display_data_info(data):
    print("Data Info:")
    print(data.info())
    print("\nData Head:")
    print(data.head())

# Preprocess the data
def preprocess_data(data):
    # Example preprocessing steps
    # Remove any rows with missing values
    data = data.dropna()
    
    # Additional preprocessing steps can be added here
    return data

if __name__ == "__main__":
    display_data_info(data)
    processed_data = preprocess_data(data)
    # Save the processed data if needed
    processed_data.to_csv('processed_training_turkish_2.csv', index=False)
    print("Data preprocessing complete.")