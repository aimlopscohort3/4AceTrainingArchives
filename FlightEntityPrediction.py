!pip -q install datasets

import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import torch
from sklearn.model_selection import train_test_split
# Load the CSV data
csv_file = "/content/flight_number_entity.csv"
data = pd.read_csv(csv_file)



# Preprocess entities from the string format
import ast  # To safely evaluate entity strings
data["entities"] = data["entities"].apply(ast.literal_eval)

# Initialize tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)  # 3 labels: O, B-FLIGHT, I-FLIGHT

# Label mapping
label2id = {"O": 0, "B-FLIGHT": 1, "I-FLIGHT": 2}
id2label = {v: k for k, v in label2id.items()}

# Data preprocessing function
def preprocess_data(example):
    tokenized_inputs = tokenizer(example["text"], truncation=True, padding="max_length", max_length=32, return_offsets_mapping=True)

    labels = ["O"] * len(tokenized_inputs["input_ids"])
    for entity in example["entities"]:
        for idx, (start, end) in enumerate(tokenized_inputs["offset_mapping"]):
            if start >= entity["start"] and end <= entity["end"]:
                labels[idx] = entity["label"]

    tokenized_inputs["labels"] = [label2id[label] for label in labels]
    tokenized_inputs.pop("offset_mapping")  # Remove offset mapping
    return tokenized_inputs

# Convert the DataFrame into a Hugging Face Dataset
dataset = Dataset.from_pandas(data)
tokenized_dataset = dataset.map(preprocess_data)

# Split into training and evaluation sets (80-20 split)
dataset_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]


import os
os.environ['WANDB_MODE'] = 'disabled'


# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add evaluation dataset
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./flight_ner_model")
tokenizer.save_pretrained("./flight_ner_model")


import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# Detect the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("./flight_ner_model")
model = BertForTokenClassification.from_pretrained("./flight_ner_model", num_labels=3)  # Example: 3 labels
model.to(device)  # Ensure model is on the same device as the input tensors

# Inference function
def predict_flight_number(text):
    # Tokenize the inputs
    tokenized_inputs = tokenizer(text, truncation=True, padding="max_length", max_length=32, return_tensors="pt")

    # Move tokenized inputs to the same device as the model
    tokenized_inputs = {key: tensor.to(device) for key, tensor in tokenized_inputs.items()}

    # Forward pass through the model
    outputs = model(**tokenized_inputs)

    # Get predictions
    predictions = torch.argmax(outputs.logits, dim=2)

    # Convert token ids to actual tokens
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])

    # Get the labels for each token (using predictions)
    labels = [id2label[label_id.item()] for label_id in predictions[0]]

    # Extract the flight number based on label
    flight_number_tokens = [token for token, label in zip(tokens, labels) if label in ["B-FLIGHT", "I-FLIGHT"]]

    return "".join(flight_number_tokens).replace("##", "")  # Remove subword token prefixes (e.g., '##')

# Example inference
test_sentence = "has my aircraft AE7898 departed?"
flight_number = predict_flight_number(test_sentence)
print(f"Extracted Flight Number: {flight_number}")
