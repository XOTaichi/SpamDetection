import pandas as pd

# Load the datasets
data2 = pd.read_csv("raw_dataset/data2.csv")
data3 = pd.read_csv("raw_dataset/data3.csv")
data4 = pd.read_csv("raw_dataset/data5.csv")

# Combine the datasets
combined_data = pd.concat([data2, data3, data4], ignore_index=True)

# Ensure the label counts are balanced
label_0 = combined_data[combined_data['label'] == 0]
label_1 = combined_data[combined_data['label'] == 1]

# Get the minimum count between label 0 and label 1
min_count = min(len(label_0), len(label_1))
print(min_count)

# Sample the same number of rows for both labels
balanced_label_0 = label_0.sample(len(label_0), random_state=42)
balanced_label_1 = label_1.sample(len(label_1), random_state=42)

# Combine the balanced data
balanced_data = pd.concat([balanced_label_0, balanced_label_1], ignore_index=True)

# Shuffle the balanced data
shuffled_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train and test sets
train_size = int(0.8 * len(shuffled_data))
train_data = shuffled_data[:train_size]
test_data = shuffled_data[train_size:]

# Save to CSV files
train_data.to_csv("dataset/body/train.csv", index=False)
test_data.to_csv("dataset/body/test.csv", index=False)
