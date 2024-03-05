# Implementation of Pre-trained CNN models using transfer learning for
# classification/object detections.
# a) AlexNet
# b) VGG-16


import tensorflow as tf
from tensorflow.keras.applications import AlexNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.alexnet import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Load the pre-trained AlexNet model without the top (classification) layer
base_model = AlexNet(weights='imagenet', include_top=False, input_shape=(227, 227, 3))

# Add a custom top (classification) layer for CIFAR-10
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
