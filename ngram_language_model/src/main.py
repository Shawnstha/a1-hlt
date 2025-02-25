# main.py

from preprocess import load_data, preprocess_data
from ngram_model import NGramModel
from smoothing import laplace_smoothing, add_k_smoothing
from perplexity import compute_perplexity

# Load and preprocess data
train_data = load_data('../data/train.txt')
validation_data = load_data('../data/val.txt')

processed_train_data = preprocess_data(train_data)
processed_validation_data = preprocess_data(validation_data)

# Train Unigram and Bigram models
unigram_model = NGramModel(n=1)
unigram_model.train(processed_train_data)

bigram_model = NGramModel(n=2)
bigram_model.train(processed_train_data)

# Calculate Perplexity without Smoothing
unigram_perplexity = compute_perplexity(unigram_model, processed_validation_data)
bigram_perplexity = compute_perplexity(bigram_model, processed_validation_data)

print(f"Unsmoothed Unigram Perplexity: {unigram_perplexity}")
print(f"Unsmoothed Bigram Perplexity: {bigram_perplexity}")

# Calculate Perplexity with Smoothing
laplace_perplexity = compute_perplexity(bigram_model, processed_validation_data, 
                                        smoothing_function=laplace_smoothing, alpha=1.0)

add_k_perplexity = compute_perplexity(bigram_model, processed_validation_data, 
                                      smoothing_function=add_k_smoothing, k=0.5)

print(f"Laplace Smoothed Bigram Perplexity: {laplace_perplexity}")
print(f"Add-k Smoothed Bigram Perplexity: {add_k_perplexity}")
