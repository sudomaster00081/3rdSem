#  use Pre-trained CNN models for feature extraction. *
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model

# Load the VGG16 model pre-trained on ImageNet data
base_model = VGG16(weights='imagenet')
# Remove the last layer (classification layer) to use it for feature extraction
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

# Example: Load an image and extract features
img_path = '/content/image.png'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Get the features from the pre-trained model
features = model.predict(img_array)

# Now 'features' contains the extracted features for the input image
features
