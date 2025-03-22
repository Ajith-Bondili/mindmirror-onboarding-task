import re
import json
import nltk
from nltk.tokenize import word_tokenize
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

def load_journal_entries():
    with open('journals.json', 'r') as file:
        return json.load(file)

journal_entries = load_journal_entries()

def preprocess_text(text):

    text = text.strip()

    # Remove emojis and non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Remove extra whitespace between words
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'•\s*([^•\n]+)', r'\1. ', text)  # Bullet •
    text = re.sub(r'-\s+([^-\n]+)', r'\1. ', text)  # Dash bullet
    text = re.sub(r'\*\s+([^*\n]+)', r'\1. ', text) # Asterisk bullet
    text = re.sub(r'\d+\.\s+([^\n]+)', r'\1. ', text) # Numbered list

    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if line.strip(): # If line is not empty after stripping
            cleaned.append(line.strip())
    text = ' '.join(cleaned)
    
# Count tokens using NLTK's word_tokenize
    tokens = word_tokenize(text)
    token_count = len(tokens)

    max_tokens = 1024      
    # If token count exceeds max_tokens, split into segments
    if token_count >= max_tokens:
        sentences = nltk.sent_tokenize(text)
        segments = []
        current_segment = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = len(word_tokenize(sentence))
            
            if current_token_count + sentence_tokens > max_tokens and current_segment:
                segments.append(' '.join(current_segment))
                current_segment = [sentence]
                current_token_count = sentence_tokens
            else:
                current_segment.append(sentence)
                current_token_count += sentence_tokens
        
        # Add the final segment if it exists
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    else:
        # If text is below max_tokens, return it as a single-element list, this way it always returns same data type if it returns segments
        return [text]

