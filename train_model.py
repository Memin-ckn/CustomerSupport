from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Load the preprocessed data
data = pd.read_csv('processed_training_turkish_2.csv')

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
        inputs = self.tokenizer(instruction, return_tensors='pt', padding=True, truncation=True)
        labels = self.tokenizer(response, return_tensors='pt', padding=True, truncation=True).input_ids
        return {"input_ids": inputs.input_ids.flatten(), "labels": labels.flatten()}

# Initialize the dataset and dataloader
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize the model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()