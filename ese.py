import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# 1. Load Data
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

# Prepare documents and labels
documents = [(" ".join(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

texts, labels = zip(*documents)
binary_labels = [1 if l == 'pos' else 0 for l in labels]

# 2. Tokenization & Padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=100)

# 3. Model Building
model = Sequential([
    Embedding(5000, 32, input_length=100),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("RNN Model Summary:")
model.summary()
# model.fit(data, binary_labels, epochs=5) # Uncomment to train
