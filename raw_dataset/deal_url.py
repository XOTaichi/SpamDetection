
import pandas as pd

# Load the URL dataset
url_data = pd.read_csv("raw_dataset/url.csv")

# Separate the data by label
url_data['Label'] = url_data['Label'].astype(str).str.upper()

# Separate the data by label
label_true = url_data[url_data['Label'] == 'TRUE']
label_false = url_data[url_data['Label'] == 'FALSE']
if len(label_true) < 25000 or len(label_false) < 25000:
    print(len(label_false))
    print(len(label_true))
    raise ValueError("Not enough data to sample 25,000 rows for each label.")

# Select 25,000 rows for both TRUE and FALSE labels
selected_true = label_true.sample(25000, random_state=42)
selected_false = label_false.sample(25000, random_state=42)

# Combine the selected data
combined_data = pd.concat([selected_true, selected_false], ignore_index=True)

# Save the combined data to a new file
combined_data.to_csv("dataset/url/train.csv", index=False)

# Remove the selected rows from the original data
remaining_true = label_true.drop(selected_true.index)
remaining_false = label_false.drop(selected_false.index)

# Select 1,000 rows from the remaining data for both TRUE and FALSE labels
remaining_sample_true = remaining_true.sample(1000, random_state=42)
remaining_sample_false = remaining_false.sample(1000, random_state=42)

# Combine the remaining samples
remaining_combined = pd.concat([remaining_sample_true, remaining_sample_false], ignore_index=True)

# Save the remaining samples to a new file
remaining_combined.to_csv("dataset/url/test.csv", index=False)