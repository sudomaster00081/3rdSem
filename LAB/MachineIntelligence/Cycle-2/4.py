# Practicing various strategies of fine tuning

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Load the pre-trained VGG-16 model without the top (classification) layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add a custom top (classification) layer for CIFAR-10
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train only the custom top layer
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


# Assume you have trained the model with the previous code snippet

# Unfreeze the last few layers for fine-tuning
for layer in model.layers[-4:]:
    layer.trainable = True

# Recompile the model to apply the changes
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),  # Use a smaller learning rate
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Fine-tune the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


# Unfreeze all layers for fine-tuning
for layer in model.layers:
    layer.trainable = True

# Recompile the model to apply the changes
model.compile(optimizer=SGD(lr=0.00001, momentum=0.9),  # Use a smaller learning rate
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Fine-tune the entire model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
