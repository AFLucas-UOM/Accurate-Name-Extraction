import pandas as pd
from transformers import pipeline

# Load the CSV file
df = pd.read_csv("DetectedNames.csv")

# Load a pretrained NER model and tokenizer
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Function to determine if the detected name is an actual person
def is_valid_name(name):
    # Use the NER pipeline to analyze the text
    entities = ner(name)
    # Check if any recognized entity in the result is labeled as 'PERSON'
    return any(entity['entity'] == 'B-PER' or entity['entity'] == 'I-PER' for entity in entities)

# Apply the filter to keep only valid names
filtered_df = df[df['Detected Name'].apply(is_valid_name)]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv("FilteredDetectedNames.csv", index=False)

print("Filtering complete. Saved as 'FilteredDetectedNames.csv'")
