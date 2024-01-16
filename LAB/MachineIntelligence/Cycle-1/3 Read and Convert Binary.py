# Read an image and convert it into binary image using threshold.


import cv2
import matplotlib.pyplot as plt

def convert_to_binary(image_path, threshold_value=128):
    # Reading the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Applying threshold
    _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    # Display 
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Binary Image')
    plt.show()
# Main
input_image_path = "LAB\MachineIntelligence\Cycle-1\image.jpg"
threshold_value = 128
convert_to_binary(input_image_path, threshold_value)
