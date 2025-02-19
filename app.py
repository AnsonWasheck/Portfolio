import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load the semantic model (this model can be replaced with any appropriate one)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the conversation CSV file.
# Assumes the CSV has one sentence per row, with user responses on even-indexed rows (starting at 0)
# and the corresponding bot responses on the following odd-indexed row.
df = pd.read_csv('conversations.csv', header=None)

# Separate user responses (even-indexed rows) and bot responses (odd-indexed rows)
user_responses = df.iloc[::2, 0].tolist()  # every 2nd row starting from index 0
bot_responses = df.iloc[1::2, 0].tolist()   # every 2nd row starting from index 1

# Precompute embeddings for user responses for efficiency
user_embeddings = model.encode(user_responses, convert_to_tensor=True)

def get_bot_response(input_text):
    """
    Given an input text, compute its embedding, determine the best matching
    stored user response using cosine similarity, and return both the corresponding
    bot response and a debug message showing:
      - The similarity score
      - An explanation
      - The matched user sentence from the CSV.
    """
    # Compute embedding for input
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    
    # Compute cosine similarities between the input and each stored user response embedding
    cosine_scores = util.cos_sim(input_embedding, user_embeddings)[0]
    
    # Find the index with the highest similarity score
    best_match_idx = int(np.argmax(cosine_scores))
    
    # Retrieve the corresponding bot response
    best_bot_response = bot_responses[best_match_idx]
    
    # Get the similarity score and the matching user sentence
    best_score = float(cosine_scores[best_match_idx])
    matching_user_sentence = user_responses[best_match_idx]
    
    # Build the debug message
    debug_msg = (
        f"Similarity score: {best_score:.2f}. A score closer to 1.0 indicates a near-perfect semantic match.\n"
        f"The sentence which semantically was decided to be closest to yours was: \"{matching_user_sentence}\""
    )
    
    # Optionally, print debug information to the server console.
    print(f"Debug: {debug_msg}")
    
    return best_bot_response, debug_msg

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/get_response')
def get_response():
    user_query = request.args.get('q', '')
    if not user_query:
        return jsonify({'response': 'No query provided.', 'debug': 'No debug info available.'})
    
    # Get both the response and the debug message.
    response, debug = get_bot_response(user_query)
    return jsonify({'response': response, 'debug': debug})

if __name__ == "__main__":
    app.run(debug=True)
