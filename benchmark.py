import seam_carving
import cv2
import os

# Load the image
image = cv2.imread('test1.jpg')

# Perform seam carving to resize image
new_width = 1000
new_height = 1000

# Resize the image using seam carving
resized_image = seam_carving.resize(image, (new_width, new_height))

# Extract the base filename and remove the extension
base_name = os.path.splitext(os.path.basename('test1.jpg'))[0]

# Create the output name with the desired format
output_name = f"{base_name}_{new_width}x{new_height}_benchmark.jpg"

# Save the resized image
cv2.imwrite(output_name, resized_image)
print(f"Image resized to {new_width}x{new_height} and saved as {output_name}")
