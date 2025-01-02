import pandas as pd
import json

# Input and output file paths
input_file = "/home/sjtu/Workspace_lyt/llm-email-spam-detection/data/raw/custon_enron/test_data.jsonl"  # Replace with your JSONL file path
output_file = "/home/sjtu/Workspace_lyt/llm-email-spam-detection/data/processed/custom_enron/test_data.csv"  # Replace with your desired CSV file path

# Read the JSONL file line by line
data = []
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Select and rename columns to match the desired format
df = df[["text", "label"]]

# Save to CSV format
df.to_csv(output_file, index=False)

print(f"JSONL file successfully transformed and saved to {output_file}.")
