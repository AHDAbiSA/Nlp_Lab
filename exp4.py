import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Input documents
docs = []

n = int(input("Enter number of documents: "))

for i in range(n):
    doc = input(f"Enter document {i + 1}: ")
    docs.append(doc)

# Input search query
query = input("\nEnter search query: ")

# Convert documents into TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Convert query into TF-IDF vector
query_vec = vectorizer.transform([query])

# Calculate TF-IDF similarity
scores = cosine_similarity(query_vec, X)

print("\nTF-IDF Similarity Scores:")
for i, score in enumerate(scores[0]):
    print(f"Document {i + 1}: {round(score, 3)}")

# Apply LSA using Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_lsa = svd.fit_transform(X)
query_lsa = svd.transform(query_vec)

# Calculate LSA similarity
lsa_scores = cosine_similarity(query_lsa, X_lsa)

print("\nLSA Similarity Scores:")
for i, score in enumerate(lsa_scores[0]):
    print(f"Document {i + 1}: {round(score, 3)}")

# Find the most relevant document
best = np.argmax(lsa_scores)

print("\nMost Relevant Document:")
print(docs[best])