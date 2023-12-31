# Display the edge map of an image with any edge detection algorithm


import cv2
import matplotlib.pyplot as plt

def display_edge_map(image_path, low_threshold=50, high_threshold=150):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_img, low_threshold, high_threshold)

    # Display the original and edge map images
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Map')

    plt.show()

# Example usage:
input_image_path = "LAB\Machine ntelligence\Cycle-1\image.jpg"
display_edge_map(input_image_path)
