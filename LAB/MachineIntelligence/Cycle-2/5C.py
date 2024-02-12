# ImplementingGenerative Models:
# a) Autoencoder for image reconstruction
# b) Word Prediction using RNN
# c) Image Captioning

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load InceptionV3 model pre-trained on ImageNet data
base_model = InceptionV3(weights='imagenet')
image_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Tokenize and pad captions
captions = ["a cat playing with a ball", "a dog running in the grass"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
total_words = len(tokenizer.word_index) + 1

input_sequences = tokenizer.texts_to_sequences(captions)
max_sequence_length = max([len(seq) for seq in input_sequences])
padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')

# Generate image features and train the captioning model
def generate_image_features(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return image_model.predict(img_array)

image_features = generate_image_features('path_to_your_image.jpg')

caption_model = Sequential()
caption_model.add(Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_length))
caption_model.add(LSTM(256))
caption_model.add(Dense(total_words, activation='softmax'))

caption_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
caption_model.fit(padded_sequences[:, :-1], tf.keras.utils.to_categorical(padded_sequences[:, 1:], num_classes=total_words), epochs=10)

# Generate captions for a new image
new_image_features = generate_image_features('path_to_new_image.jpg')
predicted_caption = []

for _ in range(max_sequence_length):
    sequence = tokenizer.texts_to_sequences([predicted_caption])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_length-1, padding='post')
    predicted_word_index = np.argmax(caption_model.predict(sequence), axis=-1)
    predicted_word = tokenizer.index_word.get(predicted_word_index[0], '')
    
    if predicted_word == 'endseq':
        break
    
    predicted_caption.append(predicted_word)

predicted_caption = ' '.join(predicted_caption)
print(predicted_caption)
