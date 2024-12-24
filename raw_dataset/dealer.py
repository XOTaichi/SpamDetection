'''
该文件是处理.eml的格式为json格式的脚本
'''
import email
from email import policy
import quopri
import re
import json
import os
import random
from urllib.parse import urlparse

def preprocessURLS(body):
    """
    Extracts URLs from the email body and removes them from the body.
    Args:
    - body: The email body text.
    Returns:
    - body: The email body with URLs removed.
    - urls_list: List of extracted URLs.
    """
    # Extract URLs from the email body
    urls_list = re.findall(r'http[s]?://[^\s<>"]+|www\.[^\s<>"]+', body)
    # Clean the body by removing URLs
    for url in urls_list:
        body = body.replace(url, "")
    return body, urls_list

def preprocess_email(email_content):
    """
    Preprocesses raw email content in bytes format and extracts headers, subject, body, and URLs.
    Args:
    - email_content: The raw email content in bytes.
    Returns:
    - dict: A dictionary containing headers, subject, body, and extracted URLs.
    """
    try:
        # Parse the email content
        email_message = email.message_from_bytes(email_content, policy=policy.default)

        # Extract the subject
        subject = email_message['subject'] or "NO SUBJECT"

        # Extract the email headers
        headers = email_message.items()
        header_string = "\n".join([f"{key}: {value}" for key, value in headers])

        # Extract the email body
        body = ""
        if email_message.is_multipart():
            # If the email has multiple parts (e.g., text and HTML), iterate through them
            for part in email_message.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain' or content_type == 'text/html':
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        body += part.get_payload(decode=True).decode(charset, errors="ignore")
                    except Exception as e:
                        print(f"Error decoding part: {e}")
        else:
            # If the email is not multipart, it's a single plain text message
            charset = email_message.get_content_charset() or 'utf-8'
            try:
                body = email_message.get_payload(decode=True).decode(charset, errors="ignore")
            except Exception as e:
                print(f"Error decoding email body: {e}")
                body = ""  # Fallback to empty body

        # Preprocess body for quoted-printable encoding
        if "Content-Transfer-Encoding: quoted-printable" in body:
            decoded_bytes_object = quopri.decodestring(body)
            body = decoded_bytes_object.decode("utf-8", errors="ignore")

        # Extract URLs and clean body
        body, urls_list = preprocessURLS(body)

        # Clean the body further
        body = re.sub(r" {2,}", " ", body)  # Remove duplicate spaces
        body = re.sub(r"\n{2,}", "\n", body)  # Remove duplicate newlines

        return {
            "headers": header_string,
            "subject": subject,
            "body": body,
            "urls": urls_list
        }

    except LookupError as e:
        print(f"Error processing email: {e} (skipping this email)")
        return None  # Return None to indicate an error in processing this email

def load_email_file(file_path):
    """
    Load the raw email content from a file.
    Args:
    - file_path: str, path to the email file.
    Returns:
    - bytes: The raw email content in bytes format.
    """
    try:
        with open(file_path, "rb") as f:
            email_content = f.read()
        return email_content
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def save_as_json(data, output_folder, idx):
    """
    Save the processed email data as a JSON file.
    Args:
    - data: The data to be saved (should be a dictionary).
    - output_folder: The folder where the file should be saved.
    - idx: The index for the JSON filename.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = os.path.join(output_folder, f"{idx}.json")
    
    try:
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"Saved processed email to {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

def process_email_folder(input_folder, output_train_folder, output_test_folder, train_ratio=0.8):
    """
    Process all email files in the specified input folder and distribute them into
    training and testing sets.
    Args:
    - input_folder: str, the folder containing the email files.
    - output_train_folder: str, the folder to store training emails.
    - output_test_folder: str, the folder to store testing emails.
    - train_ratio: float, the ratio of emails to go into the training set (default is 0.8).
    """
    # Get all email files in the input folder
    email_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Shuffle email files to randomize train/test split
    random.shuffle(email_files)

    # Split the files into training and testing sets
    split_index = int(len(email_files) * train_ratio)
    train_files = email_files[:split_index]
    test_files = email_files[split_index:]

    # Process and save the train files
    for idx, email_file in enumerate(train_files):
        file_path = os.path.join(input_folder, email_file)
        email_content = load_email_file(file_path)
        if email_content:
            result = preprocess_email(email_content)
            if result:  # Only save if processing was successful
                save_as_json(result, output_train_folder, f"train_{idx+1}")

    # Process and save the test files
    for idx, email_file in enumerate(test_files):
        file_path = os.path.join(input_folder, email_file)
        email_content = load_email_file(file_path)
        if email_content:
            result = preprocess_email(email_content)
            if result:  # Only save if processing was successful
                save_as_json(result, output_test_folder, f"test_{idx+1}")


input_folder = r"数据集\大作业\大作业\data\spam"
output_train_folder = r"dataset\whole\spam\train"
output_test_folder = r"dataset\whole\spam\test"

process_email_folder(input_folder, output_train_folder, output_test_folder)
