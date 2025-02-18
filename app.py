from flask import Flask, render_template, request, jsonify
import csv
import difflib
import os
import re

app = Flask(__name__)

CSV_FILE = os.path.join('data', 'conversations.csv')
UNANSWERED_CSV = os.path.join('data', 'unanswered.csv')
SIMILARITY_THRESHOLD = 0.55  # Adjust as needed

def preprocess(text):
    """
    Normalize pronouns so that they are treated as the same word.
    This function converts the text to lowercase and replaces specified words with "you".
    """
    text = text.lower()
    # Replace words which can be used interchangeably (e.g., anson, ansons, he) with "you"
    return re.sub(r'\b(anson|ansons|he)\b', 'you', text)

def load_conversations():
    if not os.path.exists(CSV_FILE):
        print("CSV file not found. Please run learning mode first.")
        return []
    with open(CSV_FILE, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # Each row is expected to be a single-element list
        lines = [row[0] for row in reader if row]
    return lines

# Load conversation data when the server starts
conversation_data = load_conversations()

def get_bot_response(user_input):
    user_input_processed = preprocess(user_input)
    best_similarity = 0.0
    best_index = None  # Index of the best matching human prompt
    # Compare only human prompt rows (even indices)
    for i in range(0, len(conversation_data), 2):
        stored_prompt = conversation_data[i]
        stored_prompt_processed = preprocess(stored_prompt)
        similarity = difflib.SequenceMatcher(None, user_input_processed, stored_prompt_processed).ratio()
        if similarity > best_similarity:
            best_similarity = similarity
            best_index = i

    if best_index is not None and best_similarity >= SIMILARITY_THRESHOLD:
        if best_index + 1 < len(conversation_data):
            return conversation_data[best_index + 1]
        else:
            return "Response not found for the best matching prompt."
    else:
        return "I don't have a suitable response yet."

def log_unanswered(user_input):
    """
    Append the user input to unanswered.csv if no response was found.
    """
    # Ensure that the data directory exists
    data_dir = os.path.dirname(UNANSWERED_CSV)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(UNANSWERED_CSV, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([user_input])
    print(f"Logged unanswered query: {user_input}")

@app.route('/')
def index():
    # Render the HTML interface
    return render_template('index.html')

@app.route('/get_response')
def get_response():
    user_query = request.args.get('q', '')
    if user_query:
        response = get_bot_response(user_query)
        # If no suitable response is found, log the query for later review.
        if response == "I don't have a suitable response yet.":
            log_unanswered(user_query)
        return jsonify({'response': response})
    return jsonify({'response': "No query provided."})

if __name__ == '__main__':
    app.run(debug=True)
