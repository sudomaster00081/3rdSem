# Read an image and display the RGB channel images separately.

import cv2
import matplotlib.pyplot as plt

def display_rgb_channels(image_path):
    
    img = cv2.imread(image_path)

    # Split the image into channels
    blue_channel, green_channel, red_channel = cv2.split(img)

    # original image and the RGB channels
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(blue_channel, cmap='gray')
    plt.title('Blue Channel')

    plt.subplot(2, 2, 3)
    plt.imshow(green_channel, cmap='gray')
    plt.title('Green Channel')

    plt.subplot(2, 2, 4)
    plt.imshow(red_channel, cmap='gray')
    plt.title('Red Channel')

    plt.show()

# Main
input_image_path = "LAB\Machineintelligence\Cycle-1\image.jpg"
display_rgb_channels(input_image_path)
