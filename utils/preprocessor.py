import pandas as pd
import re
from typing import List

class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text.strip()
    
    @staticmethod
    def prepare_training_data(csv_path: str) -> List[tuple]:
        df = pd.read_csv(csv_path)
        
        # Prepare training pairs
        training_pairs = []
        for _, row in df.iterrows():
            instruction = TextPreprocessor.clean_text(row['instruction_turkish'])
            response = TextPreprocessor.clean_text(row['response_turkish'])
            training_pairs.append((instruction, response))
        
        return training_pairs 