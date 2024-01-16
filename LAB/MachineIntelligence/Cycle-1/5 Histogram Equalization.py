# Apply histogram equalization on an image and display the resultant image.

import cv2
import matplotlib.pyplot as plt

def apply_histogram_equalization(image_path):
    # image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Applying histogram equalization
    equalized_img = cv2.equalizeHist(img)

    # Display
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(equalized_img, cmap='gray')
    plt.title('Equalized Image')

    plt.show()

# Main
input_image_path = "LAB\MachineIntelligence\Cycle-1\converted_image.jpg"
apply_histogram_equalization(input_image_path)
