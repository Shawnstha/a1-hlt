# ngram_model.py

from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n=1):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocabulary = set()

    def train(self, data):
        """Train the n-gram model with tokenized data."""
        for tokens in data:
            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                context = ngram[:-1]
                word = ngram[-1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
                self.vocabulary.add(word)

    def get_probability(self, context, word):
        """Calculate the probability of a word given its context."""
        if context not in self.ngram_counts:
            return 1 / len(self.vocabulary)  # Handling unknown contexts
        return self.ngram_counts[context][word] / self.context_counts[context]

    def get_vocabulary_size(self):
        """Return the size of the vocabulary."""
        return len(self.vocabulary)

# Example usage
if __name__ == '__main__':
    from preprocess import load_data, preprocess_data
    
    train_data = load_data('../data/train.txt')
    processed_train_data = preprocess_data(train_data)
    
    unigram_model = NGramModel(n=1)
    unigram_model.train(processed_train_data)
    
    bigram_model = NGramModel(n=2)
    bigram_model.train(processed_train_data)
    
    print(f"Unigram probability of 'the': {unigram_model.get_probability((), 'the')}")
    print(f"Bigram probability of 'students' given 'the': {bigram_model.get_probability(('the',), 'students')}")
