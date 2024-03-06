# ImplementingGenerative Models:
# a) Autoencoder for image reconstruction
# b) Word Prediction using RNN *
# c) Image Captioning

# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Extended corpus with more examples
corpus = [
    'This is a sample sentence.',
    'Word prediction using RNN is interesting.',
    'You can replace this dataset with your own data.',
    'Machine learning is a rapidly evolving field.',
    'Natural language processing is used in many applications.',
    'Deep learning models can achieve impressive results.',
    'Python is a popular programming language for data science.',
    'Recurrent Neural Networks are commonly used in sequence modeling.',
    'Artificial Intelligence is transforming various industries.',
    'Data preprocessing is a crucial step in machine learning.',
    'Image classification is a common computer vision task.',
    'Neural networks can be used for both regression and classification.',
    'Transfer learning allows models to leverage pre-trained knowledge.',
    'The importance of feature engineering in machine learning.',
    'Ensemble methods combine multiple models for better performance.',
    'Generative Adversarial Networks (GANs) generate realistic data.',
    'Support Vector Machines are effective in high-dimensional spaces.',
    'Clustering algorithms group similar data points together.',
    'Unsupervised learning explores patterns in unlabeled data.',
    'Ethical considerations in AI and machine learning.',
    'Hyperparameter tuning is essential for model optimization.',
    'Cross-validation helps assess the generalization of a model.',
    'Bias and fairness in machine learning algorithms.',
    'The impact of AI on the job market.',
    'Understanding the trade-off between bias and variance.',
]

# Tokenizing and padding the sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Creating input and output
X, y = padded_sequences[:, :-1], padded_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Building and training the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 100))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)

# Model summary
# model.summary()


