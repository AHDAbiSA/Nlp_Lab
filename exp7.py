import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.tag import hmm

# Download required resources
nltk.download('punkt')

# Accept tweet from user
tweet = input("Enter a tweet: ")

# Convert to lowercase and tokenize
tokens = nltk.word_tokenize(tweet.lower())

print("\nTokens:")
print(tokens)

# Generate N-grams
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print("\nUnigrams:")
print(unigrams)

print("\nBigrams:")
print(bigrams)

print("\nTrigrams:")
print(trigrams)

# Calculate word frequencies
fd = FreqDist(tokens)

print("\nWord Frequencies:")
for word, freq in fd.items():
    print(word, ":", freq)

# Sample HMM training data
train_data = [
    [('ai', 'NOUN'), ('improves', 'VERB'), ('technology', 'NOUN')],
    [('machine', 'NOUN'), ('learning', 'NOUN'), ('rocks', 'VERB')],
    [('python', 'NOUN'), ('is', 'VERB'), ('awesome', 'ADJ')]
]

# Train HMM model
trainer = hmm.HiddenMarkovModelTrainer()
model = trainer.train_supervised(train_data)

# Test sentence
test_sentence = ["ai", "improves", "technology"]

print("\nHMM Prediction:")
predicted_tags = model.tag(test_sentence)

for word, tag in predicted_tags:
    print(word, "->", tag)

# Comparison
print("\nComparison")
print("N-Gram Models capture word sequence frequencies.")
print("HMM predicts grammatical tags using contextual probabilities.")
print("HMM generally captures context better than simple N-Gram models.")