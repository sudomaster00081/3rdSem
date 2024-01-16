# Display the histogram of the gray scale image

import cv2
import matplotlib.pyplot as plt

def display_histogram(image_path):
    # image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Calculation of histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # Display 
    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Gray Scale Image')
    # histogram
    plt.subplot(2, 1, 2)
    plt.plot(hist, color='black')
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
# Main
input_image_path = "LAB\MachineIntelligence\Cycle-1\converted_image.jpg"
display_histogram(input_image_path)
