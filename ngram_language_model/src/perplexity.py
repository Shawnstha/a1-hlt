# perplexity.py

import math

def compute_perplexity(model, data, smoothing_function=None, **smoothing_params):
    """Calculate perplexity of the n-gram model on the validation dataset."""
    log_prob_sum = 0
    total_tokens = 0

    for tokens in data:
        tokens = ['<s>'] * (model.n - 1) + tokens + ['</s>']
        for i in range(model.n - 1, len(tokens)):
            context = tuple(tokens[i - model.n + 1:i])
            word = tokens[i]

            if smoothing_function:
                prob = smoothing_function(context, word, model, **smoothing_params)
            else:
                prob = model.get_probability(context, word)
            
            log_prob_sum += -math.log(prob) if prob > 0 else float('inf')
            total_tokens += 1

    return math.exp(log_prob_sum / total_tokens) if total_tokens > 0 else float('inf')
