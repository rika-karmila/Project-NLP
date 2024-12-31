import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
data = pd.read_csv('https://raw.githubusercontent.com/datasets/sms-spam-collection/master/data/sms.csv')
data = data.rename(columns={'v1': 'label', 'v2': 'text'})
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
with open('models/spam_model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print("Model has been trained and saved successfully!")
