from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from huggingface_hub import Repository
import torch
import os

# Ensure correct device selection for local or remote execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Specify path to your dataset
data_file = "C:/Users/kille/Desktop/Ollamajarv/data.txt"  # Adjust the file path
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Dataset not found at {data_file}")
print(f"Loading dataset from {data_file}...")

# Load dataset
dataset = load_dataset("text", data_files={"train": data_file})

# Specify model
model_name = "gpt2"  # Replace with an available model on Hugging Face
print(f"Loading model and tokenizer: {model_name}")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Set padding token to the EOS token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

print("Tokenizing the dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Check if test split exists; use train split for both train and eval if it doesn't
if "test" not in tokenized_datasets:
    print("Test split not found. Using train split for both training and evaluation.")
    tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",  # Logging directory
    save_strategy="epoch",  # Save model at the end of each epoch
    push_to_hub=False,  # We will push manually after training
    report_to="none"  # Disable default logging integrations
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model and tokenizer locally
trained_model_dir = "./trained_model"
print(f"Saving the model and tokenizer to {trained_model_dir}...")
os.makedirs(trained_model_dir, exist_ok=True)
model.save_pretrained(trained_model_dir)
tokenizer.save_pretrained(trained_model_dir)

# Push the trained model and tokenizer to Hugging Face Hub
repo_url = "kevenbazile/jarvisollama"  # Replace with your Hugging Face model repo URL
print(f"Pushing the model to the Hugging Face Hub repository: {repo_url}...")
repo = Repository(local_dir=trained_model_dir, clone_from=repo_url)
repo.git_pull()  # Ensure the local repository is up to date
repo.push_to_hub(commit_message="Upload trained GPT-2 model and tokenizer")

print(f"Model successfully uploaded to {repo_url}!")
