import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from transformers import GPT2Tokenizer GPT2LMHeadModel DataCollatorForLanguageModeling GPT2ForSequenceClassification Trainer TrainingArguments


from transformers import Trainer TrainingArguments
from datasets import Dataset

import warnings
warnings.filterwarnings('ignore')

# 1. Load and Preprocess Dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    label_mapping = {"neutral": 0 "negative": 1 "positive": 2}
    df['label'] = df['label'].map(label_mapping)
    return Dataset.from_pandas(df)

# 2. Load Dataset
data_path = "/content/airline_sentiment_dataset.csv"  # Path to your dataset
dataset = preprocess_data(data_path)


# Split dataset into train and validation
split = dataset.train_test_split(test_size=0.2)
train_dataset = split['train']
val_dataset = split['test']

print(train_dataset[:5])

import os
os.environ['WANDB_MODE'] = 'disabled'

# 3. Load GPT-2 Model and Tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name num_labels=3)

# GPT-2 expects padding tokens we enable padding
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# 4. Tokenization Function
def tokenize_function(example):
    return tokenizer(example['text'] padding='max_length' truncation=True max_length=128)

train_dataset = train_dataset.map(tokenize_function batched=True)
val_dataset = val_dataset.map(tokenize_function batched=True)

# Set the format for PyTorch
train_dataset.set_format(type='torch' columns=['input_ids' 'attention_mask' 'label'])
val_dataset.set_format(type='torch' columns=['input_ids' 'attention_mask' 'label'])

# 5. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results"           # Directory for saved model
    evaluation_strategy="epoch"     # Evaluate at end of each epoch
    learning_rate=5e-5              # Learning rate
    per_device_train_batch_size=8   # Batch size for training
    per_device_eval_batch_size=8    # Batch size for evaluation
    num_train_epochs=10              # Number of training epochs
    weight_decay=0.01               # Weight decay for regularization
    save_strategy="epoch"           # Save checkpoint at each epoch
    logging_dir='./logs'            # Directory for logs
    logging_steps=100
    push_to_hub=False               # Push model to Hugging Face Hub (optional)
    fp16=True                        # Use mixed precision (if supported)
)

# 6. Define Trainer
trainer = Trainer(
    model=model
    args=training_args
    train_dataset=train_dataset
    eval_dataset=val_dataset
    tokenizer=tokenizer
)

# 7. Train the Model
trainer.train()

# 8. Save the Model
trainer.save_model("gpt2-sentiment")
tokenizer.save_pretrained("gpt2-sentiment")





# Optional: Test the Model
# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    # Move inputs to the same device as the model
    inputs = tokenizer(text return_tensors="pt" padding=True truncation=True max_length=128).to(device)

    # Get predictions
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits dim=-1)

    # Return the sentiment label
    return ["neutral" "negative" "positive"][torch.argmax(predictions).item()]

# Example
example_text = "Dissapointing service shucks!"
print("Predicted Sentiment:" predict_sentiment(example_text))
