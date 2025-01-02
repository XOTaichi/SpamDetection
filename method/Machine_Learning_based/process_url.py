import pandas as pd

# Input and output file paths
input_file = "/home/sjtu/Workspace_lyt/llm-email-spam-detection/data/raw/custom_url/train.csv"  # Replace with your actual input file path
output_file = "/home/sjtu/Workspace_lyt/llm-email-spam-detection/data/processed/custom_url/train.csv"  # Replace with your desired output file path

# Load the original CSV
df = pd.read_csv(input_file)

# Replace 'TRUE' with 1 and 'FALSE' with 0 in all columns
df = df.replace({"TRUE": 1, "FALSE": 0})

# Save the updated CSV
df.to_csv(output_file, index=False)

print(f"Replaced TRUE/FALSE with 1/0 and saved to {output_file}.")
