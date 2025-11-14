# ============================================
# Sentiment Analysis using LSTM (Model Training)
# ============================================
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle

# Download stopwords
nltk.download('stopwords')

# Step 1: Load Dataset
df = pd.read_csv(r'C:\Users\Admin\Downloads\amazon_reviews.csv', encoding='latin-1')
df = df.rename(columns={'reviewText': 'review'})  # Adjust name if needed
df['sentiment'] = (df['overall'] >= 4).astype(int)

# Step 2: Text Cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['clean_review'] = df['review'].apply(clean_text)

# Step 3: Tokenization & Padding
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df['clean_review'])
sequences = tokenizer.texts_to_sequences(df['clean_review'])
padded = pad_sequences(sequences, maxlen=max_len, padding='post')

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    padded, df['sentiment'], test_size=0.2, random_state=42
)

# Step 5: Build Model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

# Step 6: Train Model
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# Step 7: Save Model and Tokenizer
model.save("model_lstm.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model and tokenizer saved successfully!")

