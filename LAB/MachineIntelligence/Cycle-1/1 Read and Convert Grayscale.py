# Read an image and convert it into gray scale image without using builtin function for the function

from PIL import Image

def convert_to_grayscale(image_path, output_path):
    # Open the image file
    img = Image.open(image_path)

    # Get the size of the image
    width, height = img.size

    # Create a new image with the same size
    grayscale_img = Image.new("L", (width, height))

    # Iterate through each pixel and convert to grayscale
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            r, g, b = img.getpixel((x, y))

            # Calculate the grayscale value using the formula: 0.299*R + 0.587*G + 0.114*B
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)

            # Set the grayscale value for the pixel in the new image
            grayscale_img.putpixel((x, y), gray_value)

    # Save the grayscale image
    grayscale_img.save(output_path)

    # Display the original and grayscale images
    img.show()
    grayscale_img.show()

# Example usage:
input_image_path = "LAB\Machine ntelligence\Cycle-1\image.jpg"
output_image_path = "LAB\Machine ntelligence\Cycle-1\converted_image.jpg"

convert_to_grayscale(input_image_path, output_image_path)
