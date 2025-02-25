# preprocess.py

import os
import re

def load_data(filepath):
    """Load data from a text file, returning a list of tokenized reviews."""
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

def preprocess_data(data, unknown_threshold=1):
    """Preprocesses the data by tokenizing and handling unknown words."""
    # Tokenize and preprocess the input data
    tokenized_data = []
    word_counts = {}

    # Tokenization and counting word frequencies
    for line in data:
        tokens = line.lower().split()  # Simple tokenization by spaces
        tokens = [re.sub(r'\W+', '', token) for token in tokens]  # Remove punctuation
        tokens = [token for token in tokens if token]  # Remove empty tokens
        tokenized_data.append(tokens)

        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1

    # Handle unknown words (replace rare words with <UNK>)
    preprocessed_data = []
    for tokens in tokenized_data:
        processed_tokens = [
            token if word_counts.get(token, 0) > unknown_threshold else '<UNK>'
            for token in tokens
        ]
        preprocessed_data.append(processed_tokens)

    return preprocessed_data

# Example usage
if __name__ == '__main__':
    train_data = load_data('../data/train.txt')
    validation_data = load_data('../data/val.txt')

    processed_train_data = preprocess_data(train_data)
    processed_validation_data = preprocess_data(validation_data)

    print(f"Sample preprocessed data: {processed_train_data[:2]}")
