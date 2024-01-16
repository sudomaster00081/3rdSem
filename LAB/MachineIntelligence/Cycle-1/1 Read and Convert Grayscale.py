# Read an image and convert it into gray scale image without using builtin function for the function

from PIL import Image

def convert_to_grayscale(image_path, output_path):
    img = Image.open(image_path)
    width, height = img.size
    grayscale_img = Image.new("L", (width, height))
    # Iterate through each pixel convert to grayscale
    for x in range(width):
        for y in range(height):     
            r, g, b = img.getpixel((x, y))
            # grayscale value formula: 0.299*R + 0.587*G + 0.114*B
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            grayscale_img.putpixel((x, y), gray_value)
    # Save 
    grayscale_img.save(output_path)
    img.show()
    grayscale_img.show()
# Main Function
input_image_path = "LAB\MachineIntelligence\Cycle-1\image.jpg"
output_image_path = "LAB\MachineIntelligence\Cycle-1\converted_image.jpg"
convert_to_grayscale(input_image_path, output_image_path)
