import json

import re
def extract_json_from_response(response):
    """
    Extract the JSON part from a response string.

    Args:
        response (str): The response string containing the JSON part.

    Returns:
        dict: The parsed JSON object if extraction and parsing are successful.
        str: An error message if extraction or parsing fails.
    """
    try:
        # Use regex to find the JSON block within triple backticks
        match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        else:
            return "No JSON block found in the response."
    except json.JSONDecodeError as e:
        return f"Failed to parse JSON: {e}"
    
def classify_email(model, email_input, feature_to_explain=None, url_info=None,
                   explanations_min=3, explanations_max=6,):
    res = model_classify(model, email_input, feature_to_explain, url_info, explanations_min, explanations_max)
    print(res)
    predicted_label, probability = explain_email(res)  
    return predicted_label,probability

def model_classify(model, email_input, feature_to_explain=None, url_info=None,
                    explanations_min=3, explanations_max=6,):
    # Initial Prompt
    messages = [
        {"role": "system", "content": f'''You are a cybersecurity and human-computer interaction expert that has the goal to detect
        if an email is legitimate or spam and help the user understand why a specific email is dangerous (or genuine), in order
        to make more informed decisions.
        The user will submit the email (body) optionally accompanied by information of the URLs in the email as:
        - server location;
        - VirusTotal scans reporting the number of scanners that detected the URL as harmless.

        Your goal is to output a JSON object containing:
        - The classification result (label).
        - The probability in percentage of the email being spam (0%=email is surely legitimate, 100%=email is surely spam) (spam_probability).
        - A list of persuasion principles that were applied by the alliged attacker (if any); each persuasion principle should be an object containing:
            the persuasion principle name (authority, scarcity, etc.),
            the part of the email that makes you say that persuasion principle is being applied;
            a brief rationale for each principle.
        - A list of {explanations_min} to {explanations_max} features that could indicate the danger (or legitimacy) of the email; the explanations must be understandable by users with no cybersecurity or computers expertise.
        {"" if feature_to_explain is None else 
        "You already know that one of the features that indicates that this email is dangerous is that " 
        + feature_to_explain["description"]}

        Desired format in **json form**:
        label: <spam/legit>
        spam_probability: <0-100%>
        persuasion_principles: [array of persuasion principles, each having: {{name, specific_sentences, rationale}} ]
        explanation: [array of {explanations_min}-{explanations_max} features explained]'''
        }
    ]
    # User input (email)
    
    email_prompt = f'''Email:
          """
          [BODY]
          {email_input}
          [\\BODY]
          """
          '''
    # Add the url_info if it exists
    if url_info is not None:
        email_prompt += f"""

          ######

          URL Information:
          {str(url_info)}"""

    messages.append({"role": "user", "content": email_prompt})
    #print(messages)
    classification_response = model.get_response(messages)

    return classification_response

def explain_email(classification_response:str):
    try:
        classification_response = extract_json_from_response(classification_response)
    except Exception as e:
        print("Invalid JSON format in the response:", classification_response)
        print(e)
        return "Invalid format", None

    if "label" in classification_response and "spam_probability" in classification_response:
        predicted_label = classification_response['label']
        probability = classification_response['spam_probability']
        return predicted_label, probability
    else:
        print("The response does not contain the predicted label (spam/non-spam)")
        return classification_response, ""