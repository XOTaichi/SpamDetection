import os
import pandas as pd

# Input and output file paths
input_file = "/home/sjtu/Workspace_lyt/llm-email-spam-detection/data/raw/custon_enron/combined_data.csv"  # Replace with your actual input file path
output_file = "/home/sjtu/Workspace_lyt/llm-email-spam-detection/data/processed/custom_enron/testdata.csv"  # Replace with your desired output file path

# Ensure the parent directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load the original CSV
df = pd.read_csv(input_file)

# Select and rename the columns to match the desired format
df.rename(columns={"body": "text"}, inplace=True)
df = df[["text", "label"]]

# Transform 'ham' to 0 and 'spam' to 1 in the label column
df['label'] = df['label'].replace({"ham": 0, "spam": 1})

# Save the transformed CSV
df.to_csv(output_file, index=False)

print(f"Data successfully transformed and saved to {output_file}.")
