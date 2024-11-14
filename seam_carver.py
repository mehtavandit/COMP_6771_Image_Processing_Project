import numpy as np
import cv2
from scipy import ndimage as ndi


def rotate_image(image, rotate_angle):
    if rotate_angle:
        return np.rot90(image, -1) #clockwise (to be checked)
    else:
        return np.rot90(image,1) #anti-clockwise (to be checked)

def energy(image):

    kernel = np.array([1,0,-1])

    x_gradient = ndi.convolve1d(image, kernel, axis=1, mode='wrap')
    y_gradient = ndi.convolve1d(image, kernel, axis=0, mode='wrap')

    gradient_magnitude = np.sqrt(np.square(x_gradient).sum(axis=2) + np.square(y_gradient).sum(axis=2))

    return gradient_magnitude


def resize(image, new_height, new_width):
    """
    Resize the image to new height and width using seam carving.
    :param image: The input image to resize.
    :param new_height: The target height.
    :param new_width: The target width.
    :return: The resized image.
    """
    image = image.astype(np.float64)
    current_height, current_width = image.shape[:2]

    # Function to handle both width and height resizing
    def adjust_size(image, dimension, current_dimension, seam_fn, rotate=False):
        if dimension != current_dimension:
            num_seams = abs(current_dimension - dimension)
            if rotate:  # Rotate image for height adjustment
                image = rotate_image(image, True)
            image = seam_fn(image, num_seams)
            if rotate:  # Rotate back after height adjustment
                image = rotate_image(image, False)
        return image

    # Adjust width and height using seam carving
    image = adjust_size(image, new_width, current_width, seams_removal if new_width < current_width else seams_insertion)
    image = adjust_size(image, new_height, current_height, seams_removal if new_height < current_height else seams_insertion, rotate=True)

    return np.uint8(np.clip(image, 0, 255))

def remove_seam(image, seam_idx):
    """
    Remove a vertical seam from the image given the seam indices.
    :param image: Input image
    :param seam_idx: The indices of the seam pixels to be removed (list or array)
    :return: The image after the seam has been removed
    """
    # The image dimensions
    h, w = image.shape[:2]

    # Create an empty array for the new image (1 less column)
    new_image = np.zeros((h, w - 1, 3), dtype=np.float64)

    for row in range(h):
        col = seam_idx[row]
        # Copy all columns except the one where the seam is located
        new_image[row, :, :] = np.delete(image[row, :, :], col, axis=0)

    return new_image


def insert_seam(image, seam_idx):
    """
    Insert a vertical seam into the image given the seam indices.
    :param image: Input image
    :param seam_idx: The indices of the seam pixels to be inserted (list or array)
    :return: The image after the seam has been inserted
    """
    h, w = image.shape[:2]

    # Create an empty array for the new image (1 more column)
    new_image = np.zeros((h, w + 1, 3), dtype=np.float64)

    for row in range(h):
        col = seam_idx[row]

        # Insert the pixel from the left and right of the seam
        left_pixel = image[row, col - 1] if col > 0 else image[row, col]
        right_pixel = image[row, col + 1] if col < w - 1 else image[row, col]

        # Average the pixels to create a smooth seam insertion
        new_pixel = (left_pixel + right_pixel) // 2

        # Copy pixels to the new image, adding the averaged pixel at the seam location
        new_image[row, :col, :] = image[row, :col, :]
        new_image[row, col, :] = new_pixel
        new_image[row, col + 1:, :] = image[row, col:, :]

    return new_image


def seams_removal(image, num_remove):
    """
    Remove seams from the image to reduce its size.
    :param image: The input image.
    :param num_remove: The number of seams to remove.
    :return: The image after the seams are removed.
    """
    for _ in range(num_remove):
        seam_idx = get_minimum_seam(image)  # Get the seam to remove
        image = remove_seam(image, seam_idx)  # Remove the seam
    return image


def seams_insertion(image, num_add):
    """
    Insert seams into the image to increase its size.
    :param image: The input image.
    :param num_add: The number of seams to add.
    :return: The image after the seams are added.
    """
    for _ in range(num_add):
        seam_idx = get_minimum_seam(image)  # Get the seam to insert
        image = insert_seam(image, seam_idx)  # Insert the seam
    return image


def get_minimum_seam(image):
    """
    Find the seam with the minimum energy in the image.
    :param image: The input image.
    :return: The indices of the minimum seam and the corresponding mask.
    """
    # Example: Here you would compute the seam indices (seam_idx) using your energy function
    # For now, we can assume that the seam_idx is computed by your seam carving algorithm
    # For simplicity, we're using a random seam as a placeholder
    height, width = image.shape[:2]
    image_energy = energy(image)

    backtrack = np.zeros_like(image_energy, dtype = int)

    for i in range(1, height):
        for j in range(0,width):
            if j == 0:
                index = np.argmin(image_energy[i-1, j:j+2])
                backtrack[i,j] = index + j
                min_energy = image_energy[i-1, index+j]
            else:
                index = np.argmin(image_energy[i-1, j-1:j+2])
                backtrack[i,j] = index + j - 1
                min_energy = image_energy[i-1, index + j -1]

            image_energy[i,j] += min_energy

    seam = []
    boolean_seam = np.ones((height, width), dtype=np.bool)
    j = np.argmin(image_energy[-1])
    for i in range(height-1, -1,-1):
        boolean_seam[i,j] = False
        seam.append(j)
        j = backtrack[i,j]

    seam.reverse()
    return np.array(seam)

def resize_and_save(image, new_width, new_height,im_path):
    """
    Function to resize the image to the specified dimensions and save it with a unique filename.
    :param image: The original image to resize.
    :param new_width: The target width.
    :param new_height: The target height.
    """
    try:
        # Perform resizing
        resized_image = resize(image, new_height, new_width)

        length_to_name = len(im_path) - 4
        image_name = im_path[0:length_to_name]

        # Display and save the resized image
        display_image = cv2.resize(im, (new_width, new_height))
        cv2.imshow(f"Resized Image - {new_width}x{new_height}", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Generate filename based on dimensions
        output_name = f"{image_name}_{new_width}x{new_height}.jpg"

        cv2.imwrite(output_name, resized_image)
        print(f"Image resized to {new_width}x{new_height} and saved as {output_name}")

    except ValueError as e:
        print("Error occurred:", e)


if __name__ == '__main__':

    # Ask for image name and load it
    im_path = input("Enter the file name: ")

    print("Select an option:")
    print("1. Resize image")
    print("2. Multiple size images")
    option = input("Enter 1 for resize or 2 for multi-size images: ")

    im = cv2.imread(im_path)
    if im is None:
        print("Error: Image not found!")
        exit()

    if option == "1":  # Single Resize
        # Single set of dimensions
        new_width = int(input("Enter the new width: "))
        new_height = int(input("Enter the new height: "))
        resize_and_save(im, new_width, new_height,im_path)

    elif option == "2":  # Multiple Resizes
        # Loop for multiple sizes
        while True:
            print("\nEnter the new dimensions (enter -1 to exit):")
            new_width = int(input("Enter the new width (or -1 to exit): "))
            new_height = int(input("Enter the new height (or -1 to exit): "))

            if new_width == -1:
                print("Exiting multiple resize mode.")
                break

            resize_and_save(im, new_width, new_height, im_path)

    else:
        print("Invalid option! Please enter 1 or 2.")