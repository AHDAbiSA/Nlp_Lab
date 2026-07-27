import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.tag import hmm

# Download required datasets
nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load Brown news corpus
news_sentences = brown.tagged_sents(categories='news', tagset='universal')

# Split into training and testing data

split = int(len(news_sentences) * 0.9)
train_data = news_sentences[:split]
test_data = news_sentences[split:]

# Train HMM POS Tagger
trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train_supervised(train_data)

# Evaluate accuracy
accuracy = hmm_tagger.accuracy(test_data)


# User input
text = input("\nEnter a sentence: ")

# Tokenize
tokens = word_tokenize(text)

# POS Tagging using HMM
tagged = hmm_tagger.tag(tokens)

print("\nTokens:")
print(tokens)

print("\nPOS Tags:")
for word, tag in tagged:
    print(f"{word} -> {tag}")

print("\nUniversal POS Tags")
print("NOUN -> Noun")
print("VERB -> Verb")
print("ADJ -> Adjective")
print("ADV -> Adverb")
print("PRON -> Pronoun")
print("DET -> Determiner")
print("ADP -> Preposition")
print("CONJ -> Conjunction")
print("NUM -> Number")
print("PRT -> Particle")
print(". -> Punctuation")
print("\nTotal Words:", len(tokens))





