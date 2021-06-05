# Text classification using Tensorflow

This file demonstrates the use of tensorflow functional API as well as custom model architectures for text classification.

**Import necessary libs**
```python
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
print(tf.__version__)
```
**Read file**
```python
train_df = pd.read_csv('../input/commonlitreadabilityprize/train.csv')
train_df = train_df[['excerpt', 'target']]
```
**Specify model parameters**
```python
vocab_size = 60000
embedding_dim = 256
max_length = 190
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8
```
**Pre-process**
```python
def pre_process(text):
    # To be added: More rigorous and correct forms of pre-processing in Natural Language modelling in general
    return text.replace('\n', ' ')
```

**Create Training file**
```python
articles = []
labels = []

for i,row in tqdm(train_df.iterrows()):
#     reader = csv.reader(csvfile, delimiter=',')
#     next(reader)
#     for row in reader:
    labels.append(row['target'])
    article = row['excerpt']
    for word in STOPWORDS:
        token = ' ' + word + ' '
        article = article.replace(token, ' ')
        article = article.replace(' ', ' ')
    article = pre_process(article)
    articles.append(article)
print(len(labels))
print(len(articles))

train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

print('train size: %d' % train_size)
print('# train articles: %d' % len(train_articles))
print('# train labels" %d' % len(train_labels))
print('# valid articles" %d' % len(validation_articles))
print('# valid labels" %d' % len(validation_labels))
```

**Tokenize input and pad them**
```python
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index
dict(list(word_index.items())[0:10])

# Find the reason for padding, since RNN or LSTM are flexible for variable input.
train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

train_labels = np.array(train_labels).reshape(-1,1)
validation_labels = np.array(validation_labels).reshape(-1,1)
```

**Model Defining**
```python
# Trivial Architecture
model = tf.keras.Sequential([
    # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

rmse = tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error', dtype=None)

model.compile(loss='mse', optimizer='adam', metrics=[rmse])

num_epochs = 10

history = model.fit(train_padded, train_labels, batch_size=128, epochs=num_epochs, validation_data=(validation_padded, validation_labels))
```
