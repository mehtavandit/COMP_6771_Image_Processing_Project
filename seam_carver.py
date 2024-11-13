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

    if new_width < current_width:
        # Remove seams to reduce width
        num_seams_to_remove = current_width - new_width
        image = seams_removal(image, num_seams_to_remove)

    elif new_width > current_width:
        # Insert seams to increase width
        num_seams_to_add = new_width - current_width
        image = seams_insertion(image, num_seams_to_add)

    if new_height < current_height:
        # Remove seams to reduce height (rotate the image to treat it as width)
        image = rotate_image(image, True)  # Rotate clockwise to swap width and height
        image = resize(image, new_width, new_height)  # Recursively call resize
        image = rotate_image(image, False)  # Rotate back to the original orientation

    elif new_height > current_height:
        # Insert seams to increase height (rotate the image to treat it as width)
        image = rotate_image(image, True)  # Rotate clockwise to swap width and height
        image = resize(image, new_width, new_height)  # Recursively call resize
        image = rotate_image(image, False)  # Rotate back to the original orientation

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
    new_image = np.zeros((h, w - 1, 3), dtype=np.uint8)

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
    new_image = np.zeros((h, w + 1, 3), dtype=np.uint8)

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





if __name__ == '__main__':
    print("Select an option:")
    print("1. Resize image")
    print("2. Object removal")
    option = input("Enter 1 for resize or 2 for object removal: ")

    if option == "1":  # Resize image
        # Ask for image name
        im_path = input("Enter the path to the image: ")

        im = cv2.imread(im_path)
        if im is None:
            print("Error: Image not found!")
            exit()

        new_width = int(input("Enter the new width: "))
        new_height = int(input("Enter the new height: "))

        try:
            resized_image = resize(im, new_height, new_width)
            output_name = input("Enter the output file name: ")
            resized_image_opencv = cv2.resize(im, (new_width, new_height))
            cv2.imshow("OpenCV Resized Image", resized_image_opencv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imwrite(output_name, resized_image)
            print(f"Image resized and saved as {output_name}")

        except ValueError as e:
            print("Error Occured")

    # elif option == "2":  # Object removal
    #     # Ask for image and mask paths
    #     im_path = input("Enter the path to the image: ")
    #     mask_path = input("Enter the path to the protective mask (optional, press Enter to skip): ")
    #     rmask_path = input("Enter the path to the removal mask: ")
    #
    #     # Load the image and masks
    #     im = cv2.imread(im_path)
    #     if im is None:
    #         print("Error: Image not found!")
    #         exit()
    #
    #     mask = cv2.imread(mask_path, 0) if mask_path else None
    #     rmask = cv2.imread(rmask_path, 0)
    #
    #     # Ask for visualization and horizontal seam removal options
    #     visualize = input("Do you want to visualize the seam removal process? (y/n): ").lower() == 'y'
    #     hremove = input("Remove horizontal seams for object removal? (y/n): ").lower() == 'y'
    #
    #     # Perform object removal
    #     output_name = input("Enter the output file name: ")
    #     # output = object_removal(im, rmask, mask, visualize, hremove)
    #     # cv2.imwrite(output_name, output)
    #     print(f"Object removal complete and saved as {output_name}")

    else:
        print("Invalid option! Please enter 1 or 2.")
