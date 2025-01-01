import pandas as pd

# File path
file_path = 'data/training_turkish_first5k.csv'

# Detect encoding manually (use 'windows-1254', 'iso-8859-9', or as detected)
try:
    data = pd.read_csv(file_path, encoding='windows-1254')  # Adjust encoding as needed
except UnicodeDecodeError:
    data = pd.read_csv(file_path, encoding='iso-8859-9')  # Fallback encoding

# Display basic information about the dataset
def display_data_info(data):
    print("Data Info:")
    print(data.info())
    print("\nData Head:")
    print(data.head())

# Preprocess the data
def preprocess_data(data):
    # Remove rows with missing values
    data = data.dropna()
    return data

if __name__ == "__main__":
    display_data_info(data)
    processed_data = preprocess_data(data)
    # Save the processed data with utf-8 encoding
    processed_data.to_csv('data/processed_training_turkish_first5k.csv', index=False, encoding='UTF-8-SIG')
    print("Data preprocessing complete.")