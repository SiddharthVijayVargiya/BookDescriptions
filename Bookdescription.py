'''import yaml

# Define the configuration settings
config_data = {
    "access_token": "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",
    "model_name": "meta-llama/Llama-2-7b-chat-hf",
    "data_file": "scraped_data.xlsx",
    "temperature": 0.7,
    "batch_size": 5,
    "max_length": 500
}

# Write to config.yaml file
with open("config.yaml", "w") as file:
    yaml.dump(config_data, file)

print("config.yaml has been created.")



import logging
import pandas as pd
import torch
import yaml
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Setup logging for tracking progress and errors
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")




# Load configuration settings from a YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)



# Login to Hugging Face Hub
login(token=config['access_token'])



# Load tokenizer and model with caching and specific settings
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, cache_dir="./cache")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, cache_dir="./cache")
    return model, tokenizer

# Load model and tokenizer based on user configuration
model_name = config['model_name']
model, tokenizer = load_model_and_tokenizer(model_name)



# Set up text generation pipeline
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)




# Load your scraped data
df = pd.read_excel(config['data_file'])





def generate_descriptions(titles_authors, temperature=config['temperature'], num_descriptions=1):
    prompts = [
        f"Create a vivid and engaging 5-line description of the book '{title}' by {author}. Focus only on the story, themes, and unique style, avoiding any restatement of the prompt."
        for title, author in titles_authors
    ]
    
    try:
        responses = llama_pipeline(prompts, max_length=100, num_return_sequences=num_descriptions, temperature=temperature)
        
        descriptions = []
        for i, response in enumerate(responses):
            description = response[0]['generated_text'].strip()
            # Explicitly removing any part of the prompt from the generated description
            for prompt in prompts:
                description = description.replace(prompt, "").strip()
            if not description:
                logging.warning(f"Empty description generated for prompt: {prompts[i]}")
                description = "Description unavailable."
            descriptions.append(description)
        return descriptions
    except Exception as e:
        logging.error(f"Error generating descriptions: {e}")
        return ["Error in description generation"] * len(prompts)




import time
import os
import logging

# Define a function to save the DataFrame incrementally
def save_dataframe_incrementally(df, filename):
    try:
        df.to_excel(filename, index=False)
        print(f"DataFrame saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving DataFrame: {e}")

# Function to generate descriptions
def generate_descriptions(titles_authors, temperature=config['temperature'], num_descriptions=1):
    prompts = [
        f"Create a vivid and engaging 5-line description of the book '{title}' by {author}. Focus only on the story, themes, and unique style, avoiding any restatement of the prompt."
        for title, author in titles_authors
    ]
    
    try:
        responses = llama_pipeline(prompts, max_length=100, num_return_sequences=num_descriptions, temperature=temperature)
        
        descriptions = []
        for i, response in enumerate(responses):
            description = response[0]['generated_text'].strip()
            # Explicitly removing any part of the prompt from the generated description
            for prompt in prompts:
                description = description.replace(prompt, "").strip()
            if not description:
                logging.warning(f"Empty description generated for prompt: {prompts[i]}")
                description = "Description unavailable."
            descriptions.append(description)
        return descriptions
    except Exception as e:
        logging.error(f"Error generating descriptions: {e}")
        return ["Error in description generation"] * len(prompts)

# Process descriptions in batches for efficiency
batch_size = config['batch_size']
max_entries = 1000  # Set the maximum number of entries to process
total_processed = 0  # Initialize a counter for processed entries
batch_number = 1  # Initialize batch number for filenames

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i + batch_size]
    titles_authors = zip(batch['Title'], batch['Author'])
    
    # Generate descriptions for the current batch
    descriptions = generate_descriptions(titles_authors, temperature=config['temperature'], num_descriptions=1)
    
    # Assign descriptions to a new column in the batch DataFrame
    batch['Description'] = descriptions  # Create a new column for descriptions
    
    # Print the descriptions immediately for real-time feedback
    for title, author, description in zip(batch['Title'], batch['Author'], descriptions):
        print(f"Processed: '{title}' by {author} - Description: {description}")

    logging.info(f"Processed batch {batch_number}")

    # Save the current batch to a new Excel file with sequential names
    batch_filename = f'excel{batch_number}.xlsx'
    save_dataframe_incrementally(batch, batch_filename)  # Save the current batch with the Description column
    
    # Increment the batch number for the next file
    batch_number += 1

    # Update the total processed count
    total_processed += len(batch)  # Update the total processed count

    # Check if the total processed count has reached the maximum limit
    if total_processed >= max_entries:
        logging.info("Reached the maximum entry limit of 1000. Stopping the processing.")
        break

# Optional: You can uncomment this part if you want to save a final consolidated file
# df.to_excel('scraped_data_with_descriptions.xlsx', index=False)
# logging.info("Descriptions for all books have been successfully added and saved.")
'''