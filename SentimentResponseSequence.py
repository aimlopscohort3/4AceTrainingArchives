import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, GPT2ForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer

from datasets import Dataset, DatasetDict

from transformers import Trainer, TrainingArguments

import warnings
warnings.filterwarnings('ignore')


# Load the dataset (Assuming it's saved as 'expanded_sentiment_conversational_data.csv')
df = pd.read_csv('/content/sentiment_conversational_response.csv')

# Add the sentiment to the user_input to create the training prompt
def prepare_dataset_with_sentiment(df):
    formatted_data = []
    for _, row in df.iterrows():
        sentiment = row['sentiment']
        user_input = row['user_input']
        response = row['response']

        # Format as 'sentiment: user_input' and add the corresponding response
        formatted_data.append({"text": f"{sentiment}: {user_input}", "response": response})
    return formatted_data


    formatted_data = prepare_dataset_with_sentiment(df)

    # Create a HuggingFace Dataset
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))

    # Split the dataset into training and validation sets (80% training, 20% validation)
    train_data, val_data = train_test_split(formatted_data, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))



    import os
    os.environ['WANDB_MODE'] = 'disabled'

    # Load GPT-2 model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)


    # GPT-2 expects padding tokens, we enable padding
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id


    # Step 6: Tokenize the dataset
    def tokenize_function(examples):

        # Tokenize the 'text' (input) column
        inputs = tokenizer(
            examples['text'],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize the 'response' (labels) column
        responses = tokenizer(
            examples['response'],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Add labels to the input encoding
        inputs['labels'] = responses['input_ids']
        return inputs

    # Apply tokenization to training and validation datasets
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Step 7: Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",  # Evaluate every epoch
        save_strategy="epoch",  # Save model every epoch
        load_best_model_at_end=True,  # Load the best model based on evaluation loss
        metric_for_best_model="eval_loss",
        logging_steps=10,
        max_grad_norm=1.0,
    )

    # Step 8: Initialize the Trainer
    trainer = Trainer(
        model=model,                       # the model to be trained
        args=training_args,                # training arguments
        train_dataset=tokenized_train_dataset,   # training dataset
        eval_dataset=tokenized_val_dataset,      # validation dataset
    )

    # Step 9: Train the model
    trainer.train()

    # Step 10: Save the fine-tuned model
    trainer.save_model("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")



###INFERENCE CODE

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

# Ensure the pad_token_id is set
model.config.pad_token_id = tokenizer.eos_token_id  # GPT-2 uses eos_token_id for padding
tokenizer.pad_token = tokenizer.eos_token  # Pad token is eos_token

# Function to generate a response
def generate_response(sentiment, user_input):
    # Format the input as 'sentiment: user_input'
    input_text = f"{sentiment}: {user_input}"

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)


    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],  # Input tokens
            attention_mask=inputs['attention_mask'],  # Attention mask
            max_length=128,  # Maximum length of the response
            num_return_sequences=1,  # Generate one response
            pad_token_id=model.config.pad_token_id,  # Set pad token
            eos_token_id=model.config.eos_token_id,  # Stop generation when EOS token is encountered
            no_repeat_ngram_size=3,  # Increase n-gram size to prevent repetition
            temperature=0.7,  # Control randomness (higher = more creative)
            top_k=50,  # Limit to top 50 tokens for each step of generation
            top_p=0.85,  # Use nucleus sampling to consider top p most probable tokens
            do_sample=True,  # Enable sampling to avoid deterministic output
        )

    # Decode the generated tokens back into text
    generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_response

# Example usage
sentiment = "negative"
user_input = "What a bad service"
response = generate_response(sentiment, user_input)
print(f"Generated Response: {response}")
