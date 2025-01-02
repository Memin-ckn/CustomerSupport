from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader

# Load the preprocessed data
data = pd.read_csv('data/processed_training_turkish_first5k.csv')

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instruction = self.data.iloc[idx]['instruction_turkish']
        response = self.data.iloc[idx]['response_turkish']
        inputs = self.tokenizer(instruction, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        labels = self.tokenizer(response, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100  # Replace padding token id's in labels by -100
        return {"input_ids": inputs.input_ids.flatten(), "labels": labels.flatten()}

# Initialize the dataset and dataloader
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize the model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500, 
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
#trainer.train()

# Path to the directory where the model will be saved
save_directory = './results/checkpoint-1875'

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Save the trained T5 model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
