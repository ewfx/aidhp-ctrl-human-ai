from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define the model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_save_path = "models/sentiment_model"

# Download the model and tokenizer
print("Downloading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to the specified path
print(f"Saving model and tokenizer to {model_save_path}...")
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("Model and tokenizer saved successfully!")	