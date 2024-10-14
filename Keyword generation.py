import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Ensure you have the necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt')

# Input text
text = "LinkedIn_AIHawk is a tool that automates the jobs application process on LinkedIn. Utilizing artificial intelligence, it enables users to apply for multiple job offers in an automated and personali"
# Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stop words and non-alphabetic characters
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

preprocessed_text = preprocess_text(text)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([preprocessed_text])

# Extract keywords
keywords = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()[0]

# Pair keywords with their TF-IDF scores and sort them
keyword_scores = sorted(zip(keywords, tfidf_scores), key=lambda x: x[1], reverse=True)

# Output top keywords
print("All words:", text.split( ) )
n=len(text.split( ))
top_keywords = [keyword for keyword, score in keyword_scores[:n]]
for i in range(len(top_keywords)):
  for j in range(i+1,len(top_keywords)):
    for char in top_keywords[i]:
      if top_keywords[i][char]==top_keywords[j][char]:
        if char < len(top_keywords[j]):
            if top_keywords[i][char] == top_keywords[j][char]:
                top_keywords[j] = top_keywords[j][:-2]
                break
print("Top Keywords:", top_keywords)
