import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset

df = pd.read_csv(r"Next-Word-Prediction-model\model\sentences.csv")


# Initialize the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])

# Create sequences
sequences = []
for line in df['text']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        sequences.append(n_gram_sequence)

# Pad the sequences
max_sequence_length = max(len(x) for x in sequences)
sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')

# Create predictors and label
X, y = sequences[:, :-1], sequences[:, -1]
y = np.array([0 if i not in tokenizer.word_index else tokenizer.word_index[i] for i in y])

# Convert labels to categorical
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

print(X.shape, y.shape)
