import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from transformers import BertForSequenceClassification,BertTokenizer,AutoTokenizer, AutoModelForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, GPT2ForSequenceClassification, Trainer, TrainingArguments


from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

import warnings
warnings.filterwarnings('ignore')


# Load Dataset
file_path = "/content/intent_dataset.csv"  # Path to your dataset
df = pd.read_csv(file_path)
dataset = Dataset.from_pandas(df)

# Split dataset into train and validation
split = dataset.train_test_split(test_size=0.2)
train_dataset = split['train']
val_dataset = split['test']


# Ensure the dataset has 'text' and 'intent' columns
assert 'text' in df.columns and 'intent' in df.columns, "CSV must contain 'text' and 'intent' columns."

print(train_dataset[:1])


# Step 2: Create a label-to-ID mapping
unique_intents = sorted(df['intent'].unique())
label2id = {label: idx for idx, label in enumerate(unique_intents)}
id2label = {idx: label for label, idx in label2id.items()}

# Map intents to numeric IDs
df['label'] = df['intent'].map(label2id)

# Step 3: Split the dataset into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)



# Step 4: Create Hugging Face datasets
train_data = Dataset.from_dict({'text': train_texts, 'label': train_labels})
val_data = Dataset.from_dict({'text': val_texts, 'label': val_labels})
dataset = DatasetDict({"train": train_data, "validation": val_data})


## Here i used this code as well for matchign with other trainings
#df = df.drop(columns=['intent'])
#dataset = Dataset.from_pandas(df)
#split = dataset.train_test_split(test_size=0.2)
#train_dataset = split['train']
#val_dataset = split['test']

# Step 5: Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(unique_intents),
                                                      label2id=label2id, id2label=id2label)


# Step 6: Tokenization function
def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch")

## Here i used this code as well for matchign with other trainings
# Apply tokenization
#tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
#tokenized_val_datasets = val_dataset.map(tokenize_function, batched=True)

#tokenized_train_datasets.set_format("torch")
#tokenized_val_datasets.set_format("torch")

# Step 7: Training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Step 8: Train the model
trainer.train()

# Step 9: Save the model and tokenizer
model.save_pretrained("./intent_model")
tokenizer.save_pretrained("./intent_model")


# Step 10: Inference example
def predict_intent(query, model, tokenizer, label_map):
    model.eval()
    inputs = tokenizer(query, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    ##Added code for device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)       # Move the model to GPU/CPU
    inputs = inputs.to(device)         # Move input tensors to GPU/CPU


    ##Added code for device
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
    return label_map[predicted_class_id]

test_query = "Can you cancel my ticket?"
predicted_intent = predict_intent(test_query, model, tokenizer, id2label)
print(f"Query: {test_query}\nPredicted Intent: {predicted_intent}")
