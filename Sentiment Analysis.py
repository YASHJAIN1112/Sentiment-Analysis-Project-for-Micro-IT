# Sentiment Analysis using Logistic Regression

# Step 1: Install Dependencies (Run in Terminal if not already installed)
# pip install nltk scikit-learn

import nltk
import random
import numpy as np
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 2: Download NLTK Data
nltk.download('movie_reviews')

# Step 3: Load and Prepare Dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

texts = [" ".join(words) for words, label in documents]
labels = [1 if label == 'pos' else 0 for words, label in documents]

# Step 4: Preprocessing - TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Step 5: Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Step 8: Test on Custom Input
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return "🙂 Positive" if prediction == 1 else "🙁 Negative"

# Test Examples
print("\n🔍 Test Examples:")
print("1:", predict_sentiment("This movie was fantastic and thrilling!"))
print("2:", predict_sentiment("It was a terrible and boring film."))
