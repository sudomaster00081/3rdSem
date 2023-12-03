#  Input an image and perform the following morphological operations
# i) Dilation
# ii) Erosion
# iii) Opening 
# iv) Closing
# Display the results.


import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the input image
image_path = "LAB\Machine ntelligence\Cycle-1\converted_image.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define a kernel for morphological operations
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Morphological Dilation
dilated_image = cv2.dilate(original_image, kernel, iterations=1)

# Morphological Erosion
eroded_image = cv2.erode(original_image, kernel, iterations=1)

# Morphological Opening (Erosion followed by Dilation)
opened_image = cv2.morphologyEx(original_image, cv2.MORPH_OPEN, kernel)

# Morphological Closing (Dilation followed by Erosion)
closed_image = cv2.morphologyEx(original_image, cv2.MORPH_CLOSE, kernel)

# Display the results
plt.figure(figsize=(10, 6))

plt.subplot(2, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilated Image')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(eroded_image, cmap='gray')
plt.title('Eroded Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(opened_image, cmap='gray')
plt.title('Opened Image')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(closed_image, cmap='gray')
plt.title('Closed Image')
plt.axis('off')

plt.tight_layout()
plt.show()
