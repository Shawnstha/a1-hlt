# smoothing.py

def laplace_smoothing(context, word, ngram_model, alpha=1.0):
    """Apply Laplace smoothing to calculate n-gram probabilities."""
    vocab_size = ngram_model.get_vocabulary_size()
    word_count = ngram_model.ngram_counts[context][word]
    context_count = ngram_model.context_counts[context]
    
    return (word_count + alpha) / (context_count + alpha * vocab_size)

def add_k_smoothing(context, word, ngram_model, k=0.5):
    """Apply Add-k smoothing to calculate n-gram probabilities."""
    vocab_size = ngram_model.get_vocabulary_size()
    word_count = ngram_model.ngram_counts[context][word]
    context_count = ngram_model.context_counts[context]
    
    return (word_count + k) / (context_count + k * vocab_size)
