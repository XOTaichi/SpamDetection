import os
import shutil
from email.parser import BytesParser
import email.policy
import random

def read_email(file_path):
    """Reads and parses an email file."""
    with open(file_path, "rb") as f:
        return BytesParser(policy=email.policy.default).parse(f)

def get_email_text(email_obj):
    """Extracts the plain text content from an email object."""
    parts = []
    for part in email_obj.walk():
        if part.get_content_type() == 'text/plain':
            parts.append(part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore'))
    return ''.join(parts)

def process_emails(folder_path):
    """Processes all email files in a folder and extracts their text content."""
    email_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    emails = []
    email_texts = []
    for file in email_files:
        try:
            email_obj = read_email(file)
            email_text = get_email_text(email_obj)
            emails.append(file)
            email_texts.append(email_text)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return emails, email_texts

def save_emails(file_paths, texts, output_folder):
    """Saves emails as text files in the specified output folder."""
    os.makedirs(output_folder, exist_ok=True)
    for idx, text in enumerate(texts):
        file_name = f"{idx}.txt"
        with open(os.path.join(output_folder, file_name), "w", encoding="utf-8") as f:
            f.write(text)

def split_and_save_emails(folder_path, output_folder):
    """Splits emails into train and test sets and saves them."""
    email_files, email_texts = process_emails(folder_path)

    # Shuffle and split the data
    combined = list(zip(email_files, email_texts))
    random.shuffle(combined)
    split_index = int(len(combined) * 0.8)
    train_set = combined[:split_index]
    test_set = combined[split_index:]

    # Save train and test sets
    save_emails(*zip(*train_set), os.path.join(output_folder, "train"))
    save_emails(*zip(*test_set), os.path.join(output_folder, "test"))

# Folder containing the emails
easy_ham_folder = "raw_dataset/easy_ham"
output_folder = "dataset/whole/ham"
spam_folder = "raw_dataset/spam"
output_folder_spam = "dataset/whole/spam"
# Process, split, and save emails
split_and_save_emails(easy_ham_folder, output_folder)
print("ham process done")
split_and_save_emails(spam_folder, output_folder_spam)
print("spam process done")