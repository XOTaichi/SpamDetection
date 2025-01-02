import csv

# Input and output file paths
input_file = "/home/sjtu/Workspace_lyt/llm-email-spam-detection/data/raw/custom_filtered_body/train_processed.csv"  # Replace with your input file path
output_file = "/home/sjtu/Workspace_lyt/llm-email-spam-detection/data/processed/custom_filtered_body/data.csv"  # Replace with your desired output file path

# Transform the data
with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
    # Read input CSV
    reader = csv.reader(infile)
    header = next(reader)  # Skip the header
    
    # Write output CSV
    writer = csv.writer(outfile)
    writer.writerow(["label", "text"])  # Write new header

    for row in reader:
        if len(row) >= 2:
            text = row[0].strip()  # First column is 'text'
            label = row[1].strip()  # Second column is 'label'

            # Ensure text is quoted if it contains commas or special characters
            text = f'"{text}"' if ',' in text or '"' in text else text

            # Write the transformed row
            writer.writerow([label, text])

print(f"Data successfully transformed and saved to {output_file}.")
